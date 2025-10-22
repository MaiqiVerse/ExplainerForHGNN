from .explainer import Explainer, ExplainerCore
from .explanation import NodeExplanationCombination, NodeExplanation
from .node_dataset_scores import node_dataset_scores
from .prepare_explanation_for_node_scores import standard_explanation
from .prepare_combined_explanation_for_node_dataset_scores import \
    prepare_combined_explanation_fn_for_node_dataset_scores
from .prepare_explanation_for_node_dataset_scores import \
    prepare_explanation_fn_for_node_dataset_scores
import torch
from tqdm import tqdm


class GradCAMCore(ExplainerCore):
    def __init__(self, config):
        super().__init__(config)

    def explain(self, model, **kwargs):
        self.model = model
        self.model.eval()

        if self.model.dataset.single_graph:
            self.node_id = kwargs.get('node_id', None)

        self.extract_neighbors_input()

        if self.model.dataset.single_graph:
            if self.node_id is None:
                raise ValueError('node_id is required for node-level explanation')
            return self.node_level_explain()
        else:
            return self.graph_level_explain()

    def init_params(self):
        pass

    def init_params_graph_level(self):
        pass

    def init_params_node_level(self):
        pass

    def graph_level_explain(self):
        pass

    def mapping_node_id(self):
        if getattr(self, 'mapped_node_id', None) is not None:
            return self.mapped_node_id
        if not self.config.get('extract_neighbors', True):
            self.mapped_node_id = self.node_id
        else:
            self.mapped_node_id = self.recovery_dict[self.node_id]
        return self.mapped_node_id

    def extract_neighbors_input(self):
        """
        Extract the neighbors of the node to be explained
        :return:
        """
        # the sample number of hencex highly depends on the number of nodes
        # Therefore, we suggests to set it to True to avoid too many samples
        if not self.config.get('extract_neighbors', True):
            gs, features = self.model.standard_input()
            self.neighbor_input = {"gs": gs, "features": features}
            return gs, features

        if getattr(self, 'neighbor_input',
                   None) is not None and self.neighbor_input.get(
            "gs", None) is not None:
            return self.neighbor_input["gs"], self.neighbor_input["features"]

        # we follow the default value in hencex
        self.n_hop = self.config.get('n_hop', 2)

        gs, features = self.model.standard_input()

        used_nodes_set = set()

        for g in gs:
            indices = g.indices()

            # consider memory-efficient
            current_nodes = [self.node_id]

            for i in range(self.n_hop):
                new_current_nodes = set()
                for node in current_nodes:
                    mask = (indices[0] == node) | (indices[1] == node)
                    used_nodes_set.update(indices[1][mask].tolist())
                    used_nodes_set.update(indices[0][mask].tolist())
                    new_current_nodes.update(indices[1][mask].tolist())
                    new_current_nodes.update(indices[0][mask].tolist())

                new_current_nodes = list(new_current_nodes)
                current_nodes = new_current_nodes

        self.used_nodes = sorted(list(used_nodes_set))
        self.recovery_dict = {node: i for i, node in enumerate(self.used_nodes)}
        self._quick_transfer = torch.zeros(len(features), dtype=torch.long
                                           ).to(self.device_string)
        for i, node in enumerate(self.used_nodes):
            self._quick_transfer[node] = i

        # now reconstruct the graph
        temp_used_nodes_tensor = torch.tensor(self.used_nodes).to(self.device_string)
        new_gs = []
        for g in gs:
            indices = g.indices()
            if self.n_hop == 1:
                # only keep the edges connected to the target node
                mask = (indices[0] == self.node_id) | (indices[1] == self.node_id)
            else:
                # we do not strict to only keep the edges in the paths between the target node and its neighbors, but keep all edges among
                # the used nodes. This is to avoid too much computation in finding the corresponding edges.
                mask = torch.isin(indices[0], temp_used_nodes_tensor) & \
                    torch.isin(indices[1], temp_used_nodes_tensor)
            # use self._quick_transfer to speed up
            new_indices = torch.stack(
                [self._quick_transfer[indices[0][mask]],
                 self._quick_transfer[indices[1][mask]]],
                dim=0)
            new_indices = new_indices.to(self.device_string)
            new_values = g.values()[mask]
            shape = torch.Size([len(self.used_nodes), len(self.used_nodes)])
            new_gs.append(torch.sparse_coo_tensor(new_indices, new_values, shape))

        self.neighbor_input = {"gs": new_gs, "features": features[self.used_nodes]}
        return self.neighbor_input["gs"], self.neighbor_input["features"]

    def node_level_explain(self):
        self.fit()
        return self.construct_explanation()

    def construct_explanation(self):
        explanation = NodeExplanation()
        explanation = standard_explanation(explanation, self)
        explanation.node_mask = self.node_mask_for_output
        for metric in self.config['eval_metrics']:
            prepare_explanation_fn_for_node_dataset_scores[metric](explanation, self)
        self.explanation = explanation
        return explanation

    def get_required_fit_params(self):
        pass

    def fit(self):
        if self.model.dataset.single_graph:
            self.fit_node_level()
        else:
            self.fit_graph_level()

    def fit_graph_level(self):
        pass

    def fit_node_level(self):
        self.set_hooks()
        loss = self.forward()
        loss.backward()
        self.remove_hooks()
        self.process_grads()
        self.calculate_cam()

    def process_grads(self):
        if self.config.get('use_meta', False):
            self.grad = [torch.mean(i, dim=0, keepdim=True) for i in self.grad]
        else:
            self.grad = torch.mean(self.grad, dim=0, keepdim=True)

    def calculate_cam(self):
        if self.config.get('use_meta', False):
            self.node_mask = [torch.relu(torch.sum(i * j, dim=1)) for i, j in
                              zip(self.grad, self.activations)]
        else:
            self.node_mask = torch.relu(torch.sum(self.grad * self.activations, dim=1))

        if self.config.get('use_feature_mask', True):
            self.feature_mask = self.node_mask
        else:
            self.edge_mask = self.convert_node_mask_to_edge_mask(self.node_mask,
                                                                 self.extract_neighbors_input()[
                                                                     0])

    def set_hooks(self):
        # if use meta, get the gradient of the output of all the last homo conv layers
        # in the last homo layers
        # if not use meta, get the gradient of the output of the last hetero conv layer

        last_hetero_conv = self.get_last_hetero_conv(self.model)

        if last_hetero_conv is None:
            raise ValueError("Cannot find the target layer")

        self.hooks = []

        if self.config.get('use_meta', False):
            last_homo_conv = self.get_last_homo_conv(last_hetero_conv)
            if last_homo_conv is None:
                raise ValueError("No enough homogeneous convolutional layer. "
                                 "GradCAM cannot find correct layer. "
                                 "Make sure that you have set is_homo_conv correctly.")
            else:
                def forward_hook(module, input, output):
                    # self.activations = output
                    if getattr(self, "activations", None) is None:
                        self.activations = []
                    self.activations.append(output)

                    def grad_hook(grad):
                        if getattr(self, "grad", None) is None:
                            self.grad = []
                        self.grad.append(grad)

                    self.hooks.append(self.activations[-1].register_hook(grad_hook))

                for homo_conv in last_homo_conv:
                    self.hooks.append(homo_conv.register_forward_hook(forward_hook))
        else:
            def forward_hook(module, input, output):
                self.activations = output

                def grad_hook(grad):
                    grad: torch.Tensor
                    self.grad = grad

                self.hooks.append(self.activations.register_hook(grad_hook))

            self.hooks.append(last_hetero_conv.register_forward_hook(forward_hook))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_last_hetero_conv(self, model):
        hetero_convs = []
        for layer in model.children():
            if getattr(layer, "is_hetero_conv", False):
                hetero_convs.append(layer)
            result = self.get_last_hetero_conv(layer)
            if result is not None:
                hetero_convs.append(result)
        if len(hetero_convs) == 0:
            return None
        return hetero_convs[-1]

    def get_homo_conv(self, model):
        homo_convs = []
        for layer in model.children():
            if getattr(layer, "is_homo_conv", False):
                homo_convs.append(layer)
            homo_convs.extend(self.get_homo_conv(layer))
        return homo_convs

    def get_last_homo_conv(self, model):
        homo_convs = self.get_homo_conv(model)
        # assume that
        # 1. only one homo conv for one meta-path (like HAN)
        # otherwise, it will directly raise error
        # if len(homo_convs) == 0:
        #     return None
        if len(homo_convs) < len(self.extract_neighbors_input()[0]):
            raise ValueError("No enough homogeneous convolutional layer. "
                             "GradCAM cannot find correct layer. "
                             "Make sure that you have set is_homo_conv correctly.")
        return homo_convs[-len(self.extract_neighbors_input()[0]):]

    def convert_node_mask_to_edge_mask(self, node_mask, gs):
        edge_mask = []
        for idx, g in enumerate(gs):
            if self.model.support_multi_features and self.config.get('use_meta', False):
                edge_mask.append(
                    self.convert_node_mask_to_edge_mask_single(node_mask[idx], g))
            else:
                edge_mask.append(
                    self.convert_node_mask_to_edge_mask_single(node_mask, g))
        return edge_mask

    @staticmethod
    def convert_node_mask_to_edge_mask_single(node_mask, g):
        g = g.coalesce()
        indices = g.indices()
        edge_mask = node_mask[indices[0]] * node_mask[indices[1]]
        return edge_mask

    def get_loss(self, output, mask=None):
        if self.model.dataset.single_graph:
            return self.get_loss_node_level(output, mask)
        else:
            return self.get_loss_graph_level(output, mask)

    def get_loss_graph_level(self, output, mask=None):
        pass

    def get_loss_node_level(self, output, mask=None):
        if mask is not None:
            class_id = mask
        else:
            class_id = output.argmax(-1)[self.mapping_node_id()]
        output = output[self.mapping_node_id()]
        loss = self.model.loss_fn(output, class_id)
        return loss

    def get_input_handle_fn(self):
        if self.model.dataset.single_graph:
            return self.get_input_handle_fn_node_level()
        else:
            return self.get_input_handle_fn_graph_level()

    def get_input_handle_fn_graph_level(self):
        pass

    def get_input_handle_fn_node_level(self):
        def handle_fn(model):
            # gs, features = model.standard_input()
            gs, features = self.extract_neighbors_input()
            return gs, features

        return handle_fn

    def forward(self):
        if self.model.dataset.single_graph:
            return self.forward_node_level()
        else:
            return self.forward_graph_level()

    def forward_graph_level(self):
        pass

    def forward_node_level(self):
        logits = self.model.custom_forward(self.get_input_handle_fn())
        if self.config.get('use_label', False):
            loss = self.get_loss(logits, self.model.labels[self.node_id])
        else:
            loss = self.get_loss(logits)
        return loss

    def build_optimizer(self):
        pass

    def build_scheduler(self, optimizer):
        pass

    def visualize(self):
        # !TODO: finish it, but not now
        pass

    @property
    def edge_mask_for_output(self):
        if 'edge_mask' not in self.__dict__:
            return None
        if not isinstance(self.edge_mask, list):
            return [self.edge_mask]
        return self.edge_mask

    @property
    def feature_mask_for_output(self):
        if 'feature_mask' not in self.__dict__:
            return None
        return self.feature_mask

    def get_custom_input_handle_fn(self, masked_gs=None, feature_mask=None):
        """
        Get the custom input handle function for the model.
        :return:
        """

        def handle_fn(model):
            gs, features = self.extract_neighbors_input()
            if masked_gs is not None:
                gs = [i.to(self.device_string) for i in masked_gs]
            if feature_mask is not None:
                if self.model.support_multi_features and self.config.get('use_meta',
                                                                         False):
                    features = [features * i.to(self.device_string).view(-1, 1) for i in
                                feature_mask]
                elif self.config.get('use_meta', False):
                    feature_mask_device = [
                        i.to(self.device_string).view(-1, 1) for i in feature_mask
                    ]
                    feature_mask_device = torch.stack(feature_mask_device, dim=0)
                    feature_mask_device = torch.mean(feature_mask_device, dim=0)
                    features = features * feature_mask_device.view(-1, 1)
                else:
                    feature_mask_device = feature_mask.to(self.device_string)
                    features = features * feature_mask_device.view(-1, 1)
            return gs, features

        return handle_fn

    @property
    def node_mask_for_output(self):
        if 'node_mask' not in self.__dict__:
            return None
        return self.node_mask


class GradCAM(Explainer):
    def __init__(self, config):
        super().__init__(config)

    def explain(self, model, **kwargs):
        self.model = model

        if self.model.dataset.single_graph:
            return self.node_level_explain(**kwargs)
        else:
            return self.graph_level_explain(**kwargs)

    def node_level_explain(self, **kwargs):
        result = []
        test_labels = self.model.dataset.labels[2]

        if kwargs.get('max_nodes', None) is not None \
            and kwargs.get('max_nodes') < len(test_labels):
            test_labels = test_labels[:kwargs.get('max_nodes')]

        for idx, label in tqdm(test_labels, desc='Explaining nodes'):
            explain_node = GradCAMCore(self.config)
            explain_node.to(self.device)
            explanation = explain_node.explain(self.model, node_id=idx)
            result.append(explanation)

        result = self.construct_explanation(result)

        self.result = result

        self.evaluate()

        self.save_summary()

        return self.eval_result

    def explain_selected_nodes(self, model, selected_nodes):
        self.model = model
        result = []
        pbar = tqdm(total=len(selected_nodes), desc='Explaining nodes')
        for idx in selected_nodes:
            if idx not in self.model.dataset.labels[2]:
                continue
            explain_node = GradCAMCore(self.config)
            explain_node.to(self.device)
            explanation = explain_node.explain(self.model, node_id=idx)
            result.append(explanation)
            pbar.update(1)

        result = self.construct_explanation(result)

        self.result = result

        self.evaluate()

        self.save_summary()

        return self.eval_result

    def graph_level_explain(self, **kwargs):
        pass

    def construct_explanation(self, result):
        result = NodeExplanationCombination(node_explanations=result)
        if self.config.get('control_data', None) is not None:
            result.control_data = self.config['control_data']

        return result

    def evaluate(self):
        eval_result = {}
        if self.config.get('eval_metrics', None) is not None:
            for metric in self.config['eval_metrics']:
                # node_dataset_score_explanations_combined[metric](self.result, self)
                self.result = prepare_combined_explanation_fn_for_node_dataset_scores[
                    metric](self.result, self)
                eval_result[metric] = node_dataset_scores[metric](self.result)

        self.eval_result = eval_result
        return eval_result

    def get_summary(self):
        return self.eval_result

    def save_summary(self):
        if self.config.get('summary_path', None) is not None:
            import os
            os.makedirs(os.path.dirname(self.config['summary_path']),
                        exist_ok=True)
            import json
            with open(self.config['summary_path'], 'w') as f:
                json.dump(self.eval_result, f)

    def save_explanation(self, **kwargs):
        if self.config.get('explanation_path', None) is not None:
            import os
            os.makedirs(self.config['explanation_path'],
                        exist_ok=True)
            self.result.save(self.config['explanation_path'], **kwargs)

    def core_class(self):
        return GradCAMCore


class GradCAMOriginal(GradCAM):
    def __init__(self, config):
        super().__init__(config)
        config['use_meta'] = False


class GradCAMMeta(GradCAM):
    def __init__(self, config):
        super().__init__(config)
        config['use_meta'] = True
