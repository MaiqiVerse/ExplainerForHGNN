from functools import partial

from scipy.special import result

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

import torch.nn.functional as F
import numpy as np


def mean(iterable):
    return sum(iterable) / len(iterable)


class SubgraphXCore(ExplainerCore):
    """
    SubgraphX core explainer
    """

    def __init__(self, config):
        super().__init__(config)
        self.record_metrics = self.config.get('record_metrics', None)
        if not self.record_metrics:
            self.record_metrics = ['mask_density']

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

    def graph_level_explain(self):
        pass

    def init_params(self):
        pass

    def init_params_graph_level(self):
        pass

    def init_params_node_level(self):
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
        value_func = self._prepare_value_func()
        reward_func = self._prepare_reward_func(value_func)
        results = self._mcts(reward_func, self.mapping_node_id())
        scores, best_subgraph = self._get_best_subgraph(results,
                                                        value_func,
                                                        self.mapping_node_id())

    def _prepare_value_func(self):

        if self.config.get('use_actual_label', False):
            # use the actual label
            label = self.model.labels[self.node_id]
        else:
            with torch.no_grad():
                temp = self.model.custom_forward(self.extract_neighbors_input)
                label = temp.argmax(-1)[self.mapping_node_id()]

        def value_func(gs, features):
            # use lambda to create a function that return gs and features
            handle_fn = lambda: (gs, features)
            with torch.no_grad():
                # use the model to get the prediction
                output = self.model.custom_forward(handle_fn)
                output = F.softmax(output, dim=-1)
                return output[self.mapping_node_id(), label].item()

        return value_func

    def _prepare_reward_func(self, value_func):

        if self.model.dataset.single_graph:
            return partial(self._node_level_mc_l_shapley, value_func=value_func)
        else:
            raise NotImplementedError(
                'Graph level explanation is not implemented yet')

    def _node_level_mc_l_shapley(self, coalition, value_func):
        sample_num = self.config.get('sample_num', 1000)

        # get current neighbors
        neighbors_num = len(self.used_nodes)

        subset = [i for i in range(neighbors_num) if i not in coalition]

        flag_for_separate = -1

        exclude_masks = []
        include_masks = []
        for _ in range(sample_num):
            subset_with_flag = np.array(subset + [flag_for_separate])
            subset_with_flag = np.random.permutation(subset_with_flag)
            # find the index of the flag
            flag_index = np.where(subset_with_flag == flag_for_separate)[0][0]
            selected_nodes = subset_with_flag[:flag_index]
            exclude_mask = np.zeros(neighbors_num)
            exclude_mask[selected_nodes] = 1.0
            exclude_mask[self.mapping_node_id()] = 1.0
            include_mask = exclude_mask.copy()
            include_mask[coalition] = 1.0
            exclude_masks.append(exclude_mask)
            include_masks.append(include_mask)

        marginal_contribution = self._get_marginal_contribution(
            include_masks, exclude_masks, value_func)
        return marginal_contribution.mean().item()

    def _get_marginal_contribution_single(self, include_mask, exclude_mask, value_func):
        # get the marginal contribution
        include_mask = torch.tensor(include_mask).to(self.device_string)
        exclude_mask = torch.tensor(exclude_mask).to(self.device_string)
        include_mask = include_mask.float()
        exclude_mask = exclude_mask.float()

        # get the features
        features = self.neighbor_input["features"]
        gs = self.neighbor_input["gs"]

        # get the values
        include_values = value_func(gs, features * include_mask)
        exclude_values = value_func(gs, features * exclude_mask)

        marginal_contribution = include_values - exclude_values
        return marginal_contribution

    def _get_marginal_contribution(self, include_masks, exclude_masks, value_func):
        results = []
        for include_mask, exclude_mask in zip(include_masks, exclude_masks):
            marginal_contribution = self._get_marginal_contribution_single(
                include_mask, exclude_mask, value_func)
            results.append(marginal_contribution)
        results = torch.cat(results, dim=0)
        return results
