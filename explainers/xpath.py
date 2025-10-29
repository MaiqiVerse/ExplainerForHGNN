from .explainer import Explainer, ExplainerCore
from datasets import NodeClassificationDataset
import torch
from .node_dataset_scores import node_dataset_scores
from .prepare_explanation_for_node_scores import standard_explanation
from .prepare_combined_explanation_for_node_dataset_scores import \
    prepare_combined_explanation_fn_for_node_dataset_scores
from .prepare_explanation_for_node_dataset_scores import \
    prepare_explanation_fn_for_node_dataset_scores
from .explanation import NodeExplanation, NodeExplanationCombination
import random
import torch.nn.functional as F
import numpy as np
import scipy


class XPathCore(ExplainerCore):
    def __init__(self, config):
        super().__init__(config)
        self.record_metrics = self.config.get('record_metrics', None)

    def explain(self, model, **kwargs):
        self.model = model
        self.model.eval()

        if self.model.dataset.single_graph:
            self.node_id = kwargs.get('node_id', None)

        if self.model.dataset.single_graph:
            if self.node_id is None:
                raise ValueError(
                    'node_id is required for node-level explanation')
            return self.node_level_explain()
        else:
            return self.graph_level_explain()

    def graph_level_explain(self):
        pass

    def init_params_graph_level(self):
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
        self.recovery_dict = {node: i for i,
                              node in enumerate(self.used_nodes)}
        self._quick_transfer = torch.zeros(len(features), dtype=torch.long
                                           ).to(self.device_string)
        for i, node in enumerate(self.used_nodes):
            self._quick_transfer[node] = i

        # now reconstruct the graph
        temp_used_nodes_tensor = torch.tensor(
            self.used_nodes).to(self.device_string)
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
            new_gs.append(torch.sparse_coo_tensor(
                new_indices, new_values, shape))

        self.neighbor_input = {"gs": new_gs,
                               "features": features[self.used_nodes]}
        return self.neighbor_input["gs"], self.neighbor_input["features"]

    def node_level_explain(self):
        self.fit()
        return self.construct_explanation()

    def construct_explanation(self):
        explanation = NodeExplanation()
        explanation = standard_explanation(explanation, self)
        for metric in self.config['eval_metrics']:
            prepare_explanation_fn_for_node_dataset_scores[metric](
                explanation, self)
        self.explanation = explanation
        return explanation

    def fit(self):
        if self.model.dataset.single_graph:
            self.fit_node_level()
        else:
            self.fit_graph_level()

    def fit_graph_level(self):
        pass

    def fit_node_level(self):
        self.extract_neighbors_input()

        paths = [[self.mapping_node_id()]]
        self.scored = {}
        while paths:
            new_paths = []
            for path in paths:
                last_node = path[-1]
                neighbors = self.get_neighbors_of_node(last_node)
                random.shuffle(neighbors)
                neighbors = neighbors[:self.config.get('sample_n', 10)]
                for neighbor in neighbors:
                    if neighbor not in path and len(path) < self.config.get('n_hop', 2):
                        new_paths.append(path + [neighbor])
            if not new_paths:
                break
            new_paths = self.scored_path(new_paths, self.scored)
            paths = new_paths[:self.config.get('max_paths_per_iteration', 5)]
        self.edge_mask = self.get_edge_masks_from_scored_paths(self.scored)

    def get_edge_masks_from_scored_paths(self, scored):
        scored_sorted = sorted(scored.keys(), key=lambda x: scored[x], reverse=True)
        edge_mask = [torch.zeros(len(g.values())) for g in self.extract_neighbors_input()[0]]
        for path in scored_sorted:
            edge_mask = self.update_edge_mask_with_path(edge_mask, path)
            if self.check_over_size(edge_mask):
                break
        return edge_mask
    
    def update_edge_mask_with_path(self, edge_mask, path):
        metapaths = self.model.config['meta_paths']
        for idx, metapath in enumerate(metapaths):
            if len(metapath) <= len(path):
                path_front = path[:len(metapath)]
                valid = True
                for node_id in path_front:
                    node_type = self.model.dataset.node_types[node_id]
                    if node_type != metapath[path_front.index(node_id)]:
                        valid = False
                        break
                if valid:
                    for i in range(len(path) - 1):
                        src = path[i]
                        dst = path[i + 1]
                        indices = self.extract_neighbors_input()[0][idx].indices()
                        mask = ((indices[0] == src) & (indices[1] == dst)) | \
                               ((indices[0] == dst) & (indices[1] == src))
                        edge_indices = torch.nonzero(mask).squeeze()
                        for edge_index in edge_indices:
                            edge_mask[idx][edge_index] = 1
                            
                path_end = path[-len(metapath):]
                valid = True
                for node_id in path_end:
                    node_type = self.model.dataset.node_types[node_id]
                    if node_type != metapath[path_end.index(node_id)]:
                        valid = False
                        break
                if valid:
                    for i in range(len(path) - 1):
                        src = path[i]
                        dst = path[i + 1]
                        indices = self.extract_neighbors_input()[0][idx].indices()
                        mask = ((indices[0] == src) & (indices[1] == dst)) | \
                               ((indices[0] == dst) & (indices[1] == src))
                        edge_indices = torch.nonzero(mask).squeeze()
                        for edge_index in edge_indices:
                            edge_mask[idx][edge_index] = 1
        return edge_mask

    def check_over_size(self, edge_mask):
        total_edges = sum([mask.shape[0] for mask in edge_mask])
        current_edges = sum([len(torch.nonzero(mask)) for mask in edge_mask])
        if self.config.get('max_explanation_size', None) is not None:
            percentage = self.config['max_explanation_size']
        elif self.config.get('edge_mask_hard_method', None) == 'top_k':
            percentage = self.config.get('top_k_for_edge_mask', 0.25)
        if current_edges >= percentage * total_edges:
            return True
        return False

    def scored_path(self, paths, scored):
        scored_paths = []
        for path in paths:
            path_tuple = tuple(path)
            if path_tuple in scored:
                score = scored[path_tuple]
            else:
                shadow_gs, shadow_X = self.break_path_and_clone_nodes(
                    path=path
                )

                def handle_fn(model):
                    return shadow_gs, shadow_X

                logits = self.model.custom_forward(
                    handle_fn)[self.mapping_node_id()]
                prob = F.softmax(logits, dim=0)
                pred = prob.argmax()
                orig_logits = self.model(
                    self.extract_neighbors_input()[0],
                    self.extract_neighbors_input()[1]
                )[self.mapping_node_id()]
                orig_prob = F.softmax(orig_logits, dim=0)
                orig_pred = orig_prob.argmax()
                score = orig_prob[orig_pred] - prob[orig_pred]
                score += 1 if prob.argmax() != orig_pred else -1
                scored[path_tuple] = score
            scored_paths.append((score, path))
        scored_paths.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored_paths]

    def break_path_and_clone_nodes(self, path):
        original_gs = self.create_original_graph_subgraph(self.used_nodes,
                                                          self.config.get('max_length_of_original_graph', 5))
        return self._break_path_and_clone_nodes(original_gs,
                                                path)

    def _break_path_and_clone_nodes(self, original_gs, path):
        if len(path) < 3:
            new_gs = []
            for g in original_gs:
                indices = g.indices()
                mask = ~((indices[0] == path[-2]) & (indices[1] == path[-1])
                         | (indices[0] == path[-1]) & (indices[1] == path[-2]))
                new_indices = indices[:, mask]
                new_values = g.values()[mask]
                shape = g.shape
                new_gs.append(torch.sparse_coo_tensor(
                    new_indices, new_values, shape))
            return self.construct_metapath_subgraph(new_gs)

        internal = path[1:-1]
        n_nodes = original_gs[0].shape[0]
        _clone_dict = {n_nodes + i: internal[i] for i in range(len(internal))}
        _reverse_clone_dict = {v: k for k, v in _clone_dict.items()}

        new_gs = []
        n_new = n_nodes + len(internal)

        for g in original_gs:
            indices = g.indices()
            link_between_clones = set()
            for i in range(len(internal)):
                if i > 0 and i < len(internal) - 2:
                    mask = (indices[0] == internal[i]) & (
                        indices[1] == internal[i + 1])
                    if len(torch.nonzero(mask)) > 0:
                        src = _reverse_clone_dict[internal[i - 1]]
                        dst = _reverse_clone_dict[internal[i + 1]]
                        link_between_clones.add((src, dst))

            last_clone_id = _reverse_clone_dict[internal[-1]]
            link_between_clones.add((last_clone_id, path[-1]))

            link_clones_to_original = set()
            for i in range(len(internal)):
                mask = (indices[0] == internal[i]) | (
                    indices[1] == internal[i])
                if len(torch.nonzero(mask)) > 0:
                    filtered_indices = indices[:, mask]
                    for j in range(filtered_indices.shape[1]):
                        src = filtered_indices[0, j].item()
                        dst = filtered_indices[1, j].item()
                        if src == internal[i]:
                            src = _reverse_clone_dict[internal[i]]
                        if dst == internal[i]:
                            dst = _reverse_clone_dict[internal[i]]
                        link_clones_to_original.add((src, dst))

            do_not_contain = set()
            for i in range(len(internal) - 1):
                do_not_contain.add(
                    (_reverse_clone_dict[internal[i]], _reverse_clone_dict[internal[i + 1]]))
                do_not_contain.add(
                    (_reverse_clone_dict[internal[i + 1]], _reverse_clone_dict[internal[i]]))
                do_not_contain.add(
                    (_reverse_clone_dict[internal[i]], _reverse_clone_dict[internal[i]]))
            do_not_contain.add((path[0], _reverse_clone_dict[internal[0]]))
            do_not_contain.add((_reverse_clone_dict[internal[0]], path[0]))
            do_not_contain.add((_reverse_clone_dict[internal[-1]], path[-1]))
            do_not_contain.add((path[-1], _reverse_clone_dict[internal[-1]]))

            link_clones_to_original -= do_not_contain

            remove_original_last_link = set(
                [(path[-2], path[-1]), (path[-1], path[-2])])

            indices_list = indices.cpu().tolist()
            indices_list = [(indices_list[0][i], indices_list[1][i])
                            for i in range(len(indices_list[0]))]
            indices_list = set(indices_list)
            indices_list -= remove_original_last_link
            indices_list.update(link_between_clones)
            indices_list.update(link_clones_to_original)
            new_indices = torch.tensor([[src for src, dst in indices_list],
                                        [dst for src, dst in indices_list]])
            new_values = torch.ones(
                new_indices.shape[1], dtype=g.values().dtype)
            shape = torch.Size([n_new, n_new])
            new_gs.append(torch.sparse_coo_tensor(
                new_indices, new_values, shape))

        return self.construct_metapath_subgraph(new_gs, _clone_dict)

    def construct_metapath_subgraph(self, gs, clone_dict=None):
        if self.model.__class__.__name__ in ['HAN', 'HAN_GCN']:
            if isinstance(self.model.dataset, NodeClassificationDataset):
                meta_paths = self.model.config['meta_paths']
                gs = [self._edges_to_metapath_adjacency(meta_path, gs) for meta_path in
                      meta_paths]
                tensor_gs = []
                for i in range(len(gs)):
                    g = gs[i]
                    g = g.tocoo()
                    indices = np.vstack((g.row, g.col))
                    values = g.data
                    tensor_gs.append(
                        torch.sparse_coo_tensor(
                            indices, values, g.shape).to(self.device_string)
                        .coalesce())
                features = self.model.dataset.node_features
                # add clone nodes' features
                if clone_dict is not None:
                    new_features = []
                    for i in range(len(features)):
                        new_features.append(features[i])
                    for i in range(len(clone_dict)):
                        new_features.append(features[clone_dict[i]])
                    features = torch.stack(new_features, dim=0)
                features = features.to(self.device_string)
                return tensor_gs, features
            else:
                raise NotImplementedError('Dataset type {} is not supported for now'.format(
                    self.model.dataset.__class__.__name__))
        else:
            raise NotImplementedError(
                'Model {} is not supported for now'.format(self.model.__class__.__name__))

    def _edges_to_metapath_adjacency(self, meta_path, gs):

        # convert gs (list of sparse adjacency) to scipy csr matrix
        new_gs = []
        for g in gs:
            g = g.coalesce()
            indices = g.indices().cpu().numpy()
            values = g.values().cpu().numpy()
            new_gs.append(
                scipy.sparse.csr_matrix(
                    (values, (indices[0], indices[1])), shape=g.shape)
            )
        gs = new_gs

        edges_directions = self.model.dataset.edges_directions

        if len(meta_path) <= 1:
            raise ValueError("Meta path should have at least two node types.")
        if len(meta_path) == 2:
            index = edges_directions['edge_types'].index(
                [meta_path[0], meta_path[1]])
            return gs[index]

        for i in range(1, len(meta_path)):
            index = edges_directions['edge_types'].index(
                [meta_path[i - 1], meta_path[i]])
            if i == 1:
                adj = gs[index]
            else:
                adj = adj @ gs[index]

        adj.data = np.ones_like(adj.data)
        return adj

    def get_neighbors_of_node(self, node_id):
        # we use the original graph to get the neighbors, not the sub-graph from metapath
        # self.extended_used_nodes, self.extended_recovery_dict = \
        gs = self.create_original_graph_subgraph(
            self.used_nodes, self.config.get('max_length_of_original_graph', 5))
        neighbors = set()
        for g in gs:
            indices = g.indices()
            mask = (indices[0] == node_id) | (indices[1] == node_id)
            neighbors.update(indices[1][mask].tolist())
            neighbors.update(indices[0][mask].tolist())
        if node_id in neighbors:
            neighbors.remove(node_id)
        return list(neighbors)

    def create_original_graph_subgraph(self, nodes, max_length):
        if getattr(self, 'original_subgraph', None) is not None:
            return self.original_subgraph

        dataset = self.model.dataset
        gs = dataset.edges
        paths = [[n] for n in nodes]
        for _ in range(max_length):
            new_paths = []
            for path in paths:
                last_node = path[-1]
                for r, g in enumerate(gs):
                    indices = g.indices()
                    mask = (indices[0] == last_node) | (
                        indices[1] == last_node)
                    neighbors = indices[1][mask].tolist(
                    ) + indices[0][mask].tolist()
                    for neighbor in neighbors:
                        if neighbor not in path:
                            new_paths.append(path + [neighbor])
            paths = new_paths

        filtered_paths = []
        for path in paths:
            if path[-1] in nodes:
                filtered_paths.append(path)

        self.extended_used_nodes = sorted(
            list(set([n for path in filtered_paths for n in path if n not in nodes])))
        self.extended_recovery_dict = {
            node: i + len(nodes) for i, node in enumerate(self.extended_used_nodes)}

        self.original_subgraph = []
        all_nodes = nodes + self.extended_used_nodes
        for g in gs:
            indices = g.indices()
            mask = torch.isin(indices[0], torch.tensor(all_nodes)) & \
                torch.isin(indices[1], torch.tensor(all_nodes))
            new_indices = torch.stack(
                [torch.tensor([all_nodes.index(i.item()) for i in indices[0][mask]]),
                 torch.tensor([all_nodes.index(i.item()) for i in indices[1][mask]])],
                dim=0)
            new_values = g.values()[mask]
            shape = torch.Size([len(all_nodes), len(all_nodes)])
            self.original_subgraph.append(
                torch.sparse_coo_tensor(new_indices, new_values, shape))

        return self.original_subgraph

    def get_input_handle_fn(self):
        if self.model.dataset.single_graph:
            return self.get_input_handle_fn_node_level()
        else:
            return self.get_input_handle_fn_graph_level()

    def get_input_handle_fn_node_level(self):
        pass

    def get_input_handle_fn_graph_level(self):
        pass

    def forward(self):
        if self.model.dataset.single_graph:
            return self.forward_node_level()
        else:
            return self.forward_graph_level()

    def forward_graph_level(self):
        pass

    def forward_node_level(self):
        pass

    def visualize(self):
        # !TODO: finish it, but not now
        pass

    @property
    def edge_mask_for_output(self):
        return self.edge_mask

    @property
    def feature_mask_for_output(self):
        return None

    def get_custom_input_handle_fn(self, masked_gs=None, feature_mask=None):
        """
        Get the custom input handle function for the model.
        :return:
        """

        def handle_fn(model):
            if model is None:
                model = self.model
            gs, features = self.extract_neighbors_input()
            if masked_gs is not None:
                gs = [i.to(self.device_string) for i in masked_gs]
            if feature_mask is not None:
                feature_mask_device = feature_mask.to(self.device_string)
                features = features * feature_mask_device
            return gs, features

        return handle_fn


class XPath(Explainer):
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

        for idx, label in test_labels:
            explain_node = self.core_class()(self.config)
            explain_node.to(self.device_string)
            explanation = explain_node.explain(self.model,
                                               node_id=idx)
            result.append(explanation)

        result = self.construct_explanation(result)

        self.result = result

        self.evaluate()

        self.save_summary()

        return self.eval_result

    def explain_selected_nodes(self, model, selected_nodes):
        self.model = model
        result = []
        test_labels = self.model.dataset.labels[2]
        for idx, label in test_labels:
            if idx in selected_nodes:
                explain_node = self.core_class()(self.config)
                explain_node.to(self.device_string)
                explanation = explain_node.explain(self.model,
                                                   node_id=idx)
                result.append(explanation)

        result = self.construct_explanation(result)

        self.result = result

        self.evaluate()

        self.save_summary()

        return self.eval_result

    def construct_explanation(self, result):
        result = NodeExplanationCombination(node_explanations=result)
        if self.config.get('control_data', None) is not None:
            result.control_data = self.config['control_data']

        return result

    def graph_level_explain(self, **kwargs):
        pass

    def evaluate(self):
        eval_result = {}
        if self.config.get('eval_metrics', None) is not None:
            for metric in self.config['eval_metrics']:
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
        return XPathCore
