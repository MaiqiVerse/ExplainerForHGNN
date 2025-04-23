from functools import partial

from pyexpat import features

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

import math
import networkx as nx
from typing import List, Tuple


def mean(iterable):
    return sum(iterable) / len(iterable)


class SimpleMCTSNode:
    def __init__(self, coalition, parent=None):
        self.coalition = coalition
        self.parent = parent
        self.children = []

        self.N = 0
        self.W = 0
        self.P = 0

    def Q(self):
        return self.W / self.N if self.N > 0 else 0

    def U(self, total_visits, c_puct):
        return c_puct * self.P * math.sqrt(total_visits) / (1 + self.N)

    def best_child(self, c_puct):
        total_visits = sum(child.N for child in self.children) + 1
        return max(self.children, key=lambda child: child.Q() + child.U(total_visits, c_puct))


class SimpleMCTS:
    def __init__(self, graph: List[nx.Graph], target_node: int, score_func,
                 c_puct=10.0, min_size=5, rollout_limit=20, coalition_max_size=None
                 ):
        self.full_graph = graph
        self.target_node = target_node
        self.score_func = score_func
        self.c_puct = c_puct
        self.min_size = min_size
        self.rollout_limit = rollout_limit

        self.visited_states = {}

        self.root = SimpleMCTSNode([list(i.nodes()) for i in self.full_graph])
        self._register_node(self.root)
        self.coalition_max_size = coalition_max_size

    def _key(self, coalition):
        result_str = ''
        for i in coalition:
            result_str += str(sorted(i))
        return result_str

    def _register_node(self, node):
        self.visited_states[self._key(node.coalition)] = node

    def expand(self, node):
        expand_nodes = self._filter_by_degree(node.coalition)

        for (i, idx) in expand_nodes:
            new_coalition = self._exclude_node(node.coalition, i, idx)
            subgraph = self._get_subgraph(new_coalition)

            check_result, final_coalition = self._check_connected(subgraph)
            if not check_result:
                continue

            if self.check_coalition_size(final_coalition):
                key = self._key(final_coalition)
                if key in self.visited_states:
                    new_node = self.visited_states[key]
                else:
                    new_node = SimpleMCTSNode(final_coalition, parent=node)
                    new_node.P = self.score_func(final_coalition)
                    self._register_node(new_node)

                if new_node not in node.children:
                    node.children.append(new_node)

    def _check_connected(self, graphs):
        results = []
        check_result = True
        for i in graphs:
            for c in nx.connected_components(i):
                if self.target_node in c:
                    results.append(c)
            else:
                check_result = False
        return check_result, results

    def _filter_by_degree(self, coalition):
        if self.coalition_max_size is None:
            return list(self._iterate_coalition(coalition))
        all_nodes = list(self._iterate_coalition(coalition))
        result = [[] for _ in range(len(coalition))]
        subgraph = self._get_subgraph(coalition)
        for i, idx in all_nodes:
            degree = self._get_degree(subgraph, (i, idx))
            if degree > 0:
                result[i].append(((i, idx), degree))

        for i in range(len(result)):
            result[i] = sorted(result[i], key=lambda x: x[1], reverse=True)
            result[i] = [x[0] for x in result[i]]
            if len(result[i]) > self.coalition_max_size:
                result[i] = result[i][:self.coalition_max_size]
        result = [x[0] for x in result]
        return result

    def _get_subgraph(self, coalition):
        result = []
        for index, i in enumerate(coalition):
            tmp = self.full_graph[index].subgraph(i)
            result.append(tmp)
        return result

    def _get_degree(self, graphs: List[nx.Graph], position: Tuple[int, int]):
        graph: nx.Graph = graphs[position[0]]
        return graph.degree[position[1]]  # type: ignore

    def _iterate_coalition(self, coalition):
        for i in range(len(coalition)):
            for j in coalition[i]:
                yield (i, j)

    def _exclude_node(self, coalition, i, idx):
        new_coalition = []
        for j in range(len(coalition)):
            if j != i:
                new_coalition.append(coalition[j])
            else:
                new_coalition.append([x for x in coalition[j] if x != idx])
        return new_coalition

    def simulate(self, node):
        return node.P

    def backpropagate(self, node, value):
        node.N += 1
        node.W += value

    def rollout(self, node=None):
        if node is None:
            node = self.root

        if not self.check_coalition_size(node.coalition):
            return self.simulate(node)

        if not node.children:
            self.expand(node)

        if node.children:
            node_next = node.best_child(self.c_puct)
            value = self.rollout(node_next)
        else:
            value = self.simulate(node)

        self.backpropagate(node, value)
        return value

    def check_coalition_size(self, coalition):
        for i in coalition:
            if len(i) <= self.min_size:
                return False
        return True

    def run(self):
        for _ in range(self.rollout_limit):
            self.rollout()

    def get_explained_nodes(self):
        return sorted(self.collect_all_nodes(), key=lambda n: n.P, reverse=True)

    def get_explanation_with_max_ratio(self, ratio):
        all_nodes = self.collect_all_nodes()
        all_nodes = sorted(all_nodes, key=lambda n: n.P, reverse=True)
        selected_node = None
        for i in range(len(all_nodes)):
            check = [
                len(all_nodes[i].coalition[j]) / len(self.full_graph[i]) < ratio
                for j in range(len(all_nodes[i].coalition))
            ]
            if all(check):
                selected_node = all_nodes[i]
                break
        if selected_node is None:
            raise ValueError('No node with coalition size less than max_ratio')
        return selected_node.coalition

    def collect_all_nodes(self):
        stack = [self.root]
        all_nodes = []
        while stack:
            node = stack.pop()
            all_nodes.append(node)
            stack.extend(node.children)
        return all_nodes


class SubgraphXCore(ExplainerCore):
    """
    SubgraphX core explainer
    """

    def __init__(self, config):
        super().__init__(config)
        self.record_metrics = self.config.get('record_metrics', None)
        if not self.record_metrics:
            self.record_metrics = ['mask_density']

        if 'fidelity_curve_auc_explanation' in self.config.get('eval_metrics', ()):
            raise ValueError(
                'SubgraphX does not support multiple thresholds. If you want to use '
                'this metric, run it in a loop with different ratio of max_nodes, and '
                'then calculate the AUC by yourself.')

        if 'graph_exp_faith_feature' in self.config.get('eval_metrics', ()):
            raise ValueError(
                'SubgraphX does not support graph_exp_faith_feature, '
                'since it is not a soft mask method. '
            )

        if 'graph_exp_faith_edge' in self.config.get('eval_metrics', ()):
            raise ValueError(
                'SubgraphX does not support graph_exp_faith_edge, '
                'since it is not a edge mask method. '
            )

        if self.config.get("feature_mask_hard_method", "top_k") != "top_k":
            raise ValueError(
                'SubgraphX only support top_k for feature mask hard method. '
            )

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
        # self.model.support_multi_features and self.config.get('use_meta', False):
        # Currently, we only support meta path version so we do not need configuration on 'use_meta'
        if not self.model.support_multi_features:
            raise ValueError('Only support meta path version, you can expect the '
                             'original version in the future release')
        value_func = self._prepare_value_func()
        reward_func = self._prepare_reward_func(value_func)
        self._mcts(reward_func)
        self.generate_node_mask()

    def generate_node_mask(self):
        # get the explanation
        selected_node = self.mcts_tree.get_explanation_with_max_ratio(
            self.config.get('top_k_for_feature_mask', 0.25))
        selected_node = selected_node.coalition
        selected_node = [sorted(i) for i in selected_node]
        _, features = self.extract_neighbors_input()
        features_list = []
        for i in range(len(selected_node)):
            features_list.append(
                torch.zeros_like(features[i]).to(self.device_string))
            features_list[i][selected_node[i]] = 1
        self.feature_mask = features_list

    def _mcts(self, reward_func):
        graphs = []
        for g in self.extract_neighbors_input()[0]:
            nx_graph = g.to_dense().cpu().numpy()
            nx_graph = nx.from_numpy_array(nx_graph)
            nx_graph = nx_graph.subgraph([
                i for i in list(nx.connected_components(nx_graph)) if self.mapping_node_id() in i
            ][0])
            graphs.append(nx_graph)
        self.mcts_tree = SimpleMCTS(
            graphs, self.mapping_node_id(), reward_func,  # type: ignore
            c_puct=self.config.get('c_puct', 10.0),
            min_size=self.config.get('min_size', 5),
            rollout_limit=self.config.get('rollout_limit', 20))
        self.mcts_tree.run()

    def _prepare_value_func(self):

        if self.config.get('use_actual_label', False):
            # use the actual label
            label = self.model.labels[self.node_id]
        else:
            gs_tmp, features_tmp = self.extract_neighbors_input()
            handle_fn = lambda model: (gs_tmp, features_tmp)
            with torch.no_grad():
                temp = self.model.custom_forward(handle_fn)
                label = temp.argmax(-1)[self.mapping_node_id()]

        def value_func(gs, features):
            # use lambda to create a function that return gs and features
            handle_fn = lambda model: (gs, features)
            with torch.no_grad():
                # use the model to get the prediction
                output = self.model.custom_forward(handle_fn)
                output = F.softmax(output, dim=-1)
                return output[self.mapping_node_id(), label].item()

        return value_func

    def _prepare_reward_func(self, value_func):

        if self.model.dataset.single_graph:
            return partial(
                self._node_level_mc_l_shapley,
                value_func=value_func)
        else:
            raise NotImplementedError(
                'Graph level explanation is not implemented yet')

    def _node_level_mc_l_shapley(self, coalition, value_func):
        sample_num = self.config.get('sample_num', 1000)

        # get current neighbors
        neighbors_num = len(self.used_nodes)

        all_masks = []

        for coalition_single_graph in coalition:
            subset = [i for i in range(neighbors_num) if i not in coalition_single_graph]

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

            all_masks.append((include_masks, exclude_masks))

        return self._get_marginal_contribution(
            all_masks,
            value_func).mean().item()

    def _get_marginal_contribution_single(self, all_masks, value_func):

        include_masks = all_masks[0]
        exclude_masks = all_masks[1]

        # Because the model support multi features
        gs = self.neighbor_input["gs"]
        features = self.neighbor_input["features"]
        new_features_include = []
        new_features_exclude = []
        for i in range(len(gs)):
            include_mask = torch.tensor(include_masks[i]).to(self.device_string)
            exclude_mask = torch.tensor(exclude_masks[i]).to(self.device_string)
            include_mask = include_mask.float()
            exclude_mask = exclude_mask.float()

            new_features_include.append(features * include_mask)
            new_features_exclude.append(features * exclude_mask)
        include_values = value_func(gs, new_features_include)
        exclude_values = value_func(gs, new_features_exclude)
        return include_values - exclude_values

    def _get_marginal_contribution(self, all_masks, value_func):
        num_mask_groups = len(all_masks[0][0])
        results = []
        for i in range(num_mask_groups):
            include_masks = []
            exclude_masks = []
            for masks in all_masks:
                include_masks.append(masks[0][i])
                exclude_masks.append(masks[1][i])
            marginal_distribution = self._get_marginal_contribution_single(
                (include_masks, exclude_masks), value_func)
            results.append(marginal_distribution)
        return torch.cat(results, dim=0)

    def visualize(self):
        # !TODO: finish it, but not now
        pass

    @property
    def edge_mask_for_output(self):
        return None

    @property
    def feature_mask_for_output(self):
        if 'feature_mask' not in self.__dict__:
            return None
        return self.feature_mask

    def get_custom_input_handle_fn(self, masked_gs=None, feature_mask=None):
        """
        Get the custom input handle function for the model.
        :param masked_gs:
        :param feature_mask:
        :return:
        """
        if masked_gs is None:
            masked_gs, _ = self.extract_neighbors_input()
        else:
            masked_gs = [i.to(self.device_string) for i in masked_gs]
        if feature_mask is None:
            feature_mask = self.feature_mask_for_output
        else:
            feature_mask = [i.to(self.device_string) for i in feature_mask]
        _, features = self.extract_neighbors_input()
        feature_mask = [
            features * i.view(-1, 1) for i in feature_mask
        ]

        return lambda model: (masked_gs, feature_mask)

    def construct_explanation(self):
        explanation = NodeExplanation()
        explanation = standard_explanation(explanation, self)
        explanation.feature_mask_hard = explanation.feature_mask
        explanation.opposite_feature_mask_hard = 1 - explanation.feature_mask_hard
        for metric in self.config['eval_metrics']:
            prepare_explanation_fn_for_node_dataset_scores[metric](explanation, self)
        self.explanation = explanation
        return explanation


class SubgraphX(Explainer):
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
            explain_node = SubgraphXCore(self.config)
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
            explain_node = SubgraphXCore(self.config)
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
