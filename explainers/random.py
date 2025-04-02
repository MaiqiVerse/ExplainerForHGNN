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


class RandomExplainerCore(ExplainerCore):
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
        if self.config.get('mask_type', "edge_mask") == "edge_mask":
            self.edge_mask = []
            for edge in self.extract_neighbors_input()[0]:
                edge = edge.coalesce()
                self.edge_mask.append(torch.rand_like(edge.values()))
        elif self.config.get('mask_type', "edge_mask") == "feature_mask":
            if self.config.get('use_meta', False) and self.model.support_multi_features:
                self.feature_mask = [torch.rand_like(self.extract_neighbors_input()[1])
                                     for _ in
                                     range(len(self.extract_neighbors_input()[0]))]
            else:
                self.feature_mask = torch.rand_like(self.extract_neighbors_input()[1])
        elif self.config.get('mask_type', "edge_mask") == "node_mask":
            if self.config.get('use_meta', False) and self.model.support_multi_features:
                self.node_mask = [
                    torch.rand_like(self.extract_neighbors_input()[1][:, 0])
                    for _ in
                    range(len(self.extract_neighbors_input()[0]))]
            else:
                self.node_mask = torch.rand_like(
                    self.extract_neighbors_input()[1][:, 0])
        elif self.config.get('mask_type', "edge_mask") == "edge_feature_mask":
            if self.config.get('use_meta', False) and self.model.support_multi_features:
                self.edge_mask = []
                self.feature_mask = []
                for edge in self.extract_neighbors_input()[0]:
                    edge = edge.coalesce()
                    self.edge_mask.append(torch.rand_like(edge.values()))
                    self.feature_mask.append(
                        torch.rand_like(self.extract_neighbors_input()[1]))
            else:
                self.edge_mask = []
                for edge in self.extract_neighbors_input()[0]:
                    edge = edge.coalesce()
                    self.edge_mask.append(torch.rand_like(edge.values()))
                self.feature_mask = torch.rand_like(self.extract_neighbors_input()[1])
        elif self.config.get('mask_type', "edge_mask") == "edge_node_mask":
            if self.config.get('use_meta', False) and self.model.support_multi_features:
                self.edge_mask = []
                self.node_mask = []
                for edge in self.extract_neighbors_input()[0]:
                    edge = edge.coalesce()
                    self.edge_mask.append(torch.rand_like(edge.values()))
                    self.node_mask.append(
                        torch.rand_like(self.extract_neighbors_input()[1][:, 0]))
            else:
                self.edge_mask = []
                for edge in self.extract_neighbors_input()[0]:
                    edge = edge.coalesce()
                    self.edge_mask.append(torch.rand_like(edge.values()))
                self.node_mask = torch.rand_like(self.extract_neighbors_input()[1][:, 0])
        else:
            raise ValueError('Invalid mask_type: {}'.format(
                self.config.get('mask_type', "edge_mask")))

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
            if 'node_mask' in self.__dict__:
                return self.node_mask
        return self.feature_mask

    def get_custom_input_handle_fn(self, masked_gs=None, feature_mask=None):
        def custom_input_handle_fn(model):
            gs, features = self.extract_neighbors_input()
            if masked_gs is not None:
                gs = masked_gs
                gs = [g.to(self.device_string) for g in gs]
            if feature_mask is not None:
                if self.config.get('mask_type', "edge_mask") == "feature_mask":
                    if self.config.get('use_meta',
                                       False) and model.support_multi_features:
                        device = self.device_string
                        feature_mask_device = [mask.to(device) for mask in feature_mask]
                        features = [features * feature_mask_device[i] for i in
                                    range(len(features))]
                    else:
                        feature_mask_device = feature_mask.to(self.device_string)
                        features = features * feature_mask_device
                elif self.config.get('mask_type', "edge_mask") == "node_mask":
                    if self.config.get('use_meta',
                                       False) and model.support_multi_features:
                        device = self.device_string
                        feature_mask_device = [mask.to(device) for mask in feature_mask]
                        features = [features * feature_mask_device[i].unsqueeze(1) for i in
                                    range(len(features))]
                    else:
                        feature_mask_device = feature_mask.to(self.device_string)
                        features = features * feature_mask_device.unsqueeze(1)
                elif self.config.get('mask_type', "edge_mask") == "edge_feature_mask":
                    if self.config.get('use_meta',
                                       False) and model.support_multi_features:
                        device = self.device_string
                        feature_mask_device = [mask.to(device) for mask in feature_mask]
                        features = [features * feature_mask_device[i] for i in
                                    range(len(features))]
                    else:
                        feature_mask_device = feature_mask.to(self.device_string)
                        features = features * feature_mask_device
                elif self.config.get('mask_type', "edge_mask") == "edge_node_mask":
                    if self.config.get('use_meta',
                                       False) and model.support_multi_features:
                        device = self.device_string
                        feature_mask_device = [mask.to(device) for mask in feature_mask]
                        features = [features * feature_mask_device[i].unsqueeze(1) for i in
                                    range(len(features))]
                    else:
                        feature_mask_device = feature_mask.to(self.device_string)
                        features = features * feature_mask_device.unsqueeze(1)
                else:
                    raise ValueError('Invalid mask_type: {}'.format(
                        self.config.get('mask_type', "edge_mask")))

            return gs, features

        return custom_input_handle_fn


class RandomExplainer(Explainer):
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
            explain_node = RandomExplainerCore(self.config)
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
            explain_node = RandomExplainerCore(self.config)
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


class RandomEdgeMaskExplainer(RandomExplainer):
    def __init__(self, config):
        super().__init__(config)
        config['mask_type'] = 'edge_mask'


class RandomFeatureMaskExplainer(RandomExplainer):
    def __init__(self, config):
        super().__init__(config)
        config['mask_type'] = 'feature_mask'


class RandomNodeMaskExplainer(RandomExplainer):
    def __init__(self, config):
        super().__init__(config)
        config['mask_type'] = 'node_mask'


class RandomEdgeAndFeatureMaskExplainer(RandomExplainer):
    def __init__(self, config):
        super().__init__(config)
        config['mask_type'] = 'edge_feature_mask'


class RandomEdgeAndNodeMaskExplainer(RandomExplainer):
    def __init__(self, config):
        super().__init__(config)
        config['mask_type'] = 'edge_node_mask'
