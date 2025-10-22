import copy
import random

import numpy as np

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

owlapy_imported = False
dlsr = None
owlapy = None

def lazy_import_cebased():
    global owlapy_imported, dlsr, owlapy
    if owlapy_imported:
        return owlapy, dlsr
    owlapy_imported = True
    try:
        import owlapy
        import owlapy.class_expression
    except ImportError:
        print("Please install owlapy to use this explainer.")
        print("You can install it by running: pip install owlapy")
        print("However, newer versions of owlapy will make"
            "owlapy.class_expression.nary_boolean_expression throw an error,\n please"
            " remove the line 21 (assert len(self._operands)>1, "
            "\"OWLNaryBooleanClassExpression requires at least one operand.\")")
        raise ImportError("owlapy is not installed.")

    from owlapy.render import DLSyntaxObjectRenderer

    dlsr = DLSyntaxObjectRenderer()
    return owlapy, dlsr


class CEBasedCore(ExplainerCore):
    def __init__(self, config):
        super().__init__(config)
        self.mother_explainer = None
        lazy_import_cebased()

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
        explanation.sparsity = self.sparsity

        explanation.feature_mask = self.feature_mask
        explanation.feature_mask_hard = self.feature_mask

        explanation.opposite_feature_mask = 1 - self.feature_mask.float()
        explanation.opposite_feature_mask_hard = 1 - self.feature_mask.float()

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
        if self.mother_explainer is None:
            raise ValueError("mother_explainer is required for node-level explanation")
        best_ce = self.mother_explainer.processing_ce[0]
        node_selected = self.mother_explainer.select_nodes_from_ce(best_ce)
        node_selected = [i for i in node_selected if i in self.used_nodes]
        original_neighbor_num = self.neighbor_input["features"].shape[0]
        self.sparsity = len(node_selected) / original_neighbor_num
        self.feature_mask = torch.zeros(original_neighbor_num, dtype=torch.bool)
        self.feature_mask = self.feature_mask.to(self.device_string)
        node_selected_mapped = [self.recovery_dict[i] for i in node_selected]
        self.feature_mask[node_selected_mapped] = True

    def get_input_handle_fn(self):
        if self.model.dataset.single_graph:
            return self.get_input_handle_fn_node_level()
        else:
            return self.get_input_handle_fn_graph_level()

    def get_input_handle_fn_graph_level(self):
        pass

    def get_input_handle_fn_node_level(self):
        pass

    def visualize(self):
        # !TODO: finish it, but not now
        pass

    @property
    def edge_mask_for_output(self):
        return None

    @property
    def feature_mask_for_output(self):
        return self.feature_mask

    def get_custom_input_handle_fn(self, masked_gs=None, feature_mask=None):
        def input_handle_fn(model):
            gs, features = self.extract_neighbors_input()
            if masked_gs is not None:
                gs = [g.to(self.device_string) for g in masked_gs]
            if feature_mask is not None:
                feature_mask_device = feature_mask.to(self.device_string)
                features = features * feature_mask_device.unsqueeze(-1).float()
            return gs, features

        return input_handle_fn


class CEBased(Explainer):
    """
    CE-based explanation method
    In some functions, we almost directly use the code from authors.
    Paper: https://arxiv.org/abs/2405.12654
    We have cited the paper. See CE-based in the README.md.
    Although we think owlapy dependency can be removed, it takes time. So we keep it.
    """
    def __init__(self, config):
        super().__init__(config)
        self.owlapy, self.dlsr = lazy_import_cebased()

    def explain(self, model, **kwargs):
        self.model = model

        if self.model.dataset.single_graph:
            return self.node_level_explain(**kwargs)
        else:
            return self.graph_level_explain(**kwargs)

    def node_level_explain(self, **kwargs):
        result = []
        test_labels = self.model.dataset.labels[2]

        self.construct_model_explanation()

        if kwargs.get('max_nodes', None) is not None \
            and kwargs.get('max_nodes') < len(test_labels):
            test_labels = test_labels[:kwargs.get('max_nodes')]

        for idx, label in tqdm(test_labels, desc='Explaining nodes'):
            explain_node = CEBasedCore(self.config)
            explain_node.mother_explainer = self
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
        self.construct_model_explanation()
        result = []
        pbar = tqdm(total=len(selected_nodes), desc='Explaining nodes')
        for idx in selected_nodes:
            if idx not in self.model.dataset.labels[2]:
                continue
            explain_node = CEBasedCore(self.config)
            explain_node.mother_explainer = self
            explain_node.to(self.device)
            explanation = explain_node.explain(self.model, node_id=idx)
            result.append(explanation)
            pbar.update(1)

        result = self.construct_explanation(result)

        self.result = result

        self.evaluate()

        self.save_summary()

        return self.eval_result

    def construct_model_explanation(self):
        self.init_beam_search()
        self.beam_search()

    def init_beam_search(self):
        self.beam_width = self.config.get('beam_width', 500)
        self.beam_depth = self.config.get('beam_depth', 15)
        # please fit it to [max(length of meta paths) - 1] * num of layers (HAN-like)
        # fit it to the number of layers of the GNN model (meta-path-free)
        self.max_depth = self.config.get('max_depth', 4)
        self.init_starting_ce()

    def init_starting_ce(self):
        target_node_type = self.model.dataset.edge_directions['target_node_type']
        from owlapy.class_expression import OWLClass
        self.processing_ce = [OWLClass('#' + str(target_node_type))] * self.beam_width

        if self.config.get('use_cache', True):
            self.cache = {}

    def beam_search(self):
        starting_ce = self.processing_ce[0]
        for i in range(self.beam_depth):
            print(f"Processing depth {i} / {self.beam_depth}")
            new_processing_ce = copy.deepcopy(self.processing_ce)
            for ce in self.processing_ce:
                new_ce = self.create_new_ce(ce)
                if new_ce != ce:
                    new_processing_ce.append(new_ce)
            scored_items = [(x, self.score(x)) for x in
                            tqdm(new_processing_ce, desc="Scoring CEs")]
            scored_items = sorted(scored_items, key=lambda x: x[1], reverse=True)
            filtered_items = [x[0] for x in scored_items if
                              x[1] > 0 and x[0] != starting_ce]
            if len(filtered_items) > self.beam_width:
                new_processing_ce = filtered_items[:self.beam_width]
            else:
                if len(filtered_items) == 0:
                    print("Warning: No new CEs are created.")
                new_processing_ce = filtered_items
                new_processing_ce = new_processing_ce + [starting_ce] * (
                    self.beam_width - len(new_processing_ce))
            self.processing_ce = new_processing_ce

    def score(self, ce):
        if self.config.get('use_cache', True) and ce in self.cache:
            return self.cache[ce]
        node_selected = self.select_nodes_from_ce(ce)
        if len(node_selected) != 0:
            score = self.get_score(node_selected)
        else:
            score = 0
        if self.config.get('use_cache', True):
            self.cache[ce] = score
        return score

    def get_score(self, node_selected):
        original_pred = self.model.forward()
        original_pred = original_pred.argmax(dim=-1)
        new_gs, new_features = self.generate_new_input(node_selected)
        new_pred = self.model.custom_forward(lambda model: (new_gs, new_features))
        new_pred = new_pred.argmax(dim=-1)
        return self.fidelity_neg_model(original_pred, new_pred)

    def fidelity_neg_model(self, original_pred, new_pred):
        test_nodes = self.model.dataset.labels[2]
        test_nodes = [node[0] for node in test_nodes]
        original_pred = original_pred[test_nodes]
        new_idx = [self.mapping[node] if node in self.mapping else -1 for node in
                   test_nodes]
        new_pred = new_pred[new_idx]
        return (original_pred == new_pred).sum().item() / len(test_nodes)

    def generate_new_input(self, node_selected):
        gs, features = self.model.standard_input()
        mask = torch.zeros(len(features), dtype=torch.bool)
        node_selected = list(sorted(node_selected))
        self.mapping = {node: i for i, node in enumerate(node_selected)}
        node_selected = torch.tensor(node_selected).to(self.device_string)
        mask[node_selected] = True
        mask = mask.to(self.device_string)
        new_features = features[mask, :]
        new_gs = []
        for i, g in enumerate(gs):
            indices = g.indices()
            mask = torch.isin(indices[0], node_selected) & torch.isin(indices[1],
                                                                      node_selected)
            new_indices = indices[:, mask]
            new_values = g.values()[mask]
            shape = torch.Size([len(node_selected), len(node_selected)])
            new_gs.append(torch.sparse_coo_tensor(new_indices, new_values, shape).to(
                self.device_string))

        return new_gs, new_features

    def select_nodes_from_ce(self, ce):
        available_node_types = list(range(max(self.model.dataset.node_types) + 1))
        top_classes = self.retrieve_top_classes(ce)
        available_node_types = [x for x in available_node_types if x in top_classes]
        # nodes = self.model.dataset.node_types
        nodes = np.isin(self.model.dataset.node_types, np.array(available_node_types))
        nodes = np.where(nodes)[0]
        nodes_selected = set()
        for node in nodes:
            if self.iterate_through_graph(ce, self.model.dataset.node_types[node],
                                          node):
                nodes_selected.add(node)
        return list(sorted(list(nodes_selected)))

    def iterate_through_graph(self, ce, current_node_type, current_id):
        if isinstance(ce, owlapy.class_expression.OWLClass):
            if current_node_type == int(dlsr.render(ce)):
                return True
            return False
        elif isinstance(ce, owlapy.class_expression.OWLObjectUnionOf):
            for op in ce.operands():
                if self.iterate_through_graph(op, current_node_type, current_id):
                    return True
            return False
        elif isinstance(ce, owlapy.class_expression.OWLObjectIntersectionOf):
            result = True
            for op in ce.operands():
                result = result and self.iterate_through_graph(
                    op, current_node_type, current_id)
            return result
        elif isinstance(ce, owlapy.class_expression.OWLObjectMinCardinality):
            new_node_types = self.retrieve_top_classes(ce._filler)
            list_adjacent_nodes = self.get_adjacent_nodes(
                current_node_type, current_id, new_node_types)
            number_trues = 0
            for node in list_adjacent_nodes:
                if self.iterate_through_graph(ce._filler,
                                              self.model.dataset.node_types[node],
                                              node):
                    number_trues += 1
                    if number_trues >= ce._cardinality:
                        return True

            return False

    def get_adjacent_nodes(self, current_node_type, current_id,
                           new_node_types
                           ):
        list_adjacent_nodes = []
        # if current_node_type in the edge one side, and new_node_types in the other side
        # we can use the code which links current node
        for index, edge_directions in enumerate(
            self.model.dataset.edge_directions['edge_types']):
            if edge_directions[0] == current_node_type and edge_directions[
                1] in new_node_types:
                edge = self.model.dataset.edges[index]
                edge = edge.tocsr()
                list_adjacent_nodes.extend(edge[current_id].indices.tolist())

        return list_adjacent_nodes

    def retrieve_top_classes(self, ce):
        if isinstance(ce, owlapy.class_expression.OWLClass):
            str_class = int(dlsr.render(ce))
            return [str_class]
        elif isinstance(ce, owlapy.class_expression.OWLObjectIntersectionOf):
            result = []
            for op in ce.operands():
                new_class = self.retrieve_top_classes(op)
                if 0 < len(new_class):
                    result.extend(new_class)
            if len(result) >= 2:
                return []
            else:
                return result
        elif isinstance(ce, owlapy.class_expression.OWLObjectUnionOf):
            result = []
            for op in ce.operands():
                result.append(op)
            return result
        return []

    def create_new_ce(self, ce):
        mutations = self.find_all_possible_mutations(ce)
        for mutation in mutations:
            new_ce = self.random_mutation(ce, mutation)
            if new_ce is not None:
                return new_ce
        raise ValueError("No new CE is created.")  # will remove it in the future

    def find_all_possible_mutations(self, ce):
        mutations = []
        if isinstance(ce, owlapy.class_expression.OWLObjectUnionOf):
            mutations.append("union")
            if len(mutations) == 4:
                return list(set(mutations))
            for op in ce.operands():
                mutations.extend(self.find_all_possible_mutations(op))
        elif isinstance(ce, owlapy.class_expression.OWLObjectIntersectionOf):
            mutations.append("intersection")
            if len(mutations) == 4:
                return list(set(mutations))
            for op in ce.operands():
                mutations.extend(self.find_all_possible_mutations(op))
        elif isinstance(ce, owlapy.class_expression.OWLObjectRestriction):
            mutations.append("cardinality")
            if len(mutations) == 4:
                return list(set(mutations))
            mutations.extend(self.find_all_possible_mutations(ce._filler))
        elif isinstance(ce, owlapy.class_expression.OWLClass):
            mutations.append("class")
            if len(mutations) == 4:
                return list(set(mutations))
        return list(set(mutations))

    def random_mutation(self, ce, mutation):
        if mutation == "class":
            total_classes = self.count_total_classes(ce)
            if total_classes == 1:
                new_ce = owlapy.class_expression.OWLObjectIntersectionOf([ce])
                return self.random_mutation(new_ce, "intersection")
            list_indices = [i for i in range(total_classes)]
            random.shuffle(list_indices)
            for i in list_indices:
                new_ce = copy.deepcopy(ce)
                if self.replace_nth_class(new_ce, i, self.new_edge()):
                    if self.get_max_depth(new_ce) <= self.max_depth:
                        return new_ce
            return None
        elif mutation == "intersection":
            total_intersections = self.count_total_intersections(ce)
            if total_intersections == 0:
                return None
            list_indices = [i for i in range(1, total_intersections + 1)]
            random.shuffle(list_indices)
            for i in list_indices:
                new_ce = copy.deepcopy(ce)
                if self.replace_nth_intersection(new_ce, i, self.new_edge()):
                    if self.get_max_depth(new_ce) <= self.max_depth:
                        return new_ce
            return None
        elif mutation == "union":
            total_unions = self.count_total_unions(ce)
            if total_unions == 0:
                return None
            list_indices = [i for i in range(1, total_unions)]
            random.shuffle(list_indices)
            for i in list_indices:
                new_ce = copy.deepcopy(ce)
                if self.replace_nth_union(new_ce, i, self.new_class()):
                    if self.get_max_depth(new_ce) <= self.max_depth:
                        return new_ce
            return None
        elif mutation == "cardinality":
            total_restrictions = self.count_total_restrictions(ce)
            if total_restrictions == 0:
                return None
            n = random.randint(1, total_restrictions)
            new_ce = copy.deepcopy(ce)
            if self.increase_nth_existential_restriction(new_ce, n):
                return new_ce
            return None
        return None

    def increase_nth_existential_restriction(self, ce, n, increase=1):
        if isinstance(ce, owlapy.class_expression.OWLCardinalityRestriction):
            if n == 1:
                ce._cardinality += increase
                return True
            n -= 1
            return self.increase_nth_existential_restriction(ce._filler, n)
        elif isinstance(ce, owlapy.class_expression.OWLNaryBooleanClassExpression):
            for i, op in enumerate(ce.operands()):
                if self.increase_nth_existential_restriction(op, n):
                    return True

    def count_total_restrictions(self, ce):
        count = 0
        if isinstance(ce, owlapy.class_expression.OWLObjectRestriction):
            count += 1
            count += self.count_total_restrictions(ce._filler)
        elif isinstance(ce, owlapy.class_expression.OWLNaryBooleanClassExpression):
            for op in ce.operands():
                count += self.count_total_restrictions(op)
        return count

    def replace_nth_union(self, ce, n, new_class_value, top_ce=None):
        if isinstance(ce, owlapy.class_expression.OWLClass):
            if n == 1:
                new_union = owlapy.class_expression.OWLObjectUnionOf(
                    [new_class_value, ce])
                new_operands = tuple(
                    x for x in top_ce._operands if x != ce
                )
                new_operands += (new_union,)
                top_ce._operands = new_operands
                return True
            n -= 1
        elif isinstance(ce, owlapy.class_expression.OWLNaryBooleanClassExpression):
            for i, op in enumerate(ce.operands()):
                if self.replace_nth_union(op, n, new_class_value, ce):
                    return True
        elif isinstance(ce, owlapy.class_expression.OWLObjectRestriction):
            return self.replace_nth_union(ce._filler, n, new_class_value, top_ce)
        return False

    def count_total_unions(self, ce):
        count = 0
        if isinstance(ce, owlapy.class_expression.OWLObjectUnionOf):
            count += 1
            for op in ce.operands():
                count += self.count_total_unions(op)
        elif isinstance(ce, owlapy.class_expression.OWLObjectIntersectionOf):
            for op in ce.operands():
                count += self.count_total_unions(op)
        elif isinstance(ce, owlapy.class_expression.OWLObjectRestriction):
            count += self.count_total_unions(ce._filler)
        return count

    def replace_nth_intersection(self, ce, n, new_property_value):
        if isinstance(ce, owlapy.class_expression.OWLObjectIntersectionOf):
            if n == 1:
                if isinstance(new_property_value,
                              owlapy.class_expression.OWLCardinalityRestriction):
                    for op in ce.operands():
                        if op == new_property_value:
                            op._cardinality += 1
                            return True
                ce._operands += (new_property_value,)
                return True
            n -= 1
            for i, op in enumerate(ce.operands()):
                self.replace_nth_intersection(op, n, new_property_value)
        elif isinstance(ce, owlapy.class_expression.OWLNaryBooleanClassExpression):
            for i, op in enumerate(ce.operands()):
                self.replace_nth_intersection(op, n, new_property_value)
        elif isinstance(ce, owlapy.class_expression.OWLObjectRestriction):
            self.replace_nth_intersection(ce._filler, n, new_property_value)
        return False

    def count_total_intersections(self, ce):
        count = 0
        if isinstance(ce, owlapy.class_expression.OWLObjectIntersectionOf):
            count += 1
            for op in ce.operands():
                count += self.count_total_intersections(op)
        elif isinstance(ce, owlapy.class_expression.OWLObjectUnionOf):
            for op in ce.operands():
                count += self.count_total_intersections(op)
        elif isinstance(ce, owlapy.class_expression.OWLObjectRestriction):
            count += self.count_total_intersections(ce._filler)
        return count

    def get_max_depth(self, ce):
        if isinstance(ce, owlapy.class_expression.OWLClass):
            return 1
        elif isinstance(ce, owlapy.class_expression.OWLNaryBooleanClassExpression):
            max_depth = 0
            for op in ce.operands():
                max_depth = max(max_depth, self.get_max_depth(op))
            return max_depth
        elif isinstance(ce, owlapy.class_expression.OWLObjectRestriction):
            return self.get_max_depth(ce._filler)
        return 0

    def new_edge(self):
        return owlapy.class_expression.OWLObjectMinCardinality(
            cardinality=1, filler=self.new_class(), property=self.new_property())

    def new_class(self):
        return random.choice(list(range(max(self.model.dataset.node_types) + 1)))

    def new_property(self):
        return "to"

    def count_total_classes(self, ce):
        if isinstance(ce, owlapy.class_expression.OWLClass):
            return 1
        elif isinstance(ce, owlapy.class_expression.OWLNaryBooleanClassExpression):
            count = 0
            for op in ce.operands():
                count += self.count_total_classes(op)
            return count
        elif isinstance(ce, owlapy.class_expression.OWLObjectRestriction):
            return self.count_total_classes(ce._filler)
        return 0

    def replace_nth_class(self, ce, n, new_property_value, top_ce=None):
        if isinstance(ce, owlapy.class_expression.OWLClass):
            if n == 1:
                if top_ce is None:
                    top_ce = owlapy.class_expression.OWLObjectIntersectionOf([ce])
                if isinstance(top_ce, owlapy.class_expression.OWLObjectIntersectionOf):
                    if isinstance(new_property_value,
                                  owlapy.class_expression.OWLCardinalityRestriction):
                        for op in top_ce.operands():
                            if op == new_property_value:
                                op._cardinality += 1
                                return True
                        top_ce._operands += (new_property_value,)
                        return True
                    elif isinstance(new_property_value,
                                    owlapy.class_expression.OWLObjectIntersectionOf):
                        top_ce._operands += (new_property_value,)
                        return True
                    print("Error: new_property_value is not a valid value.")
                    print("new_property_value is type:", type(new_property_value))
                    raise ValueError("new_property_value is not a valid value.")
                elif isinstance(top_ce, owlapy.class_expression.OWLObjectRestriction):
                    new_filler = owlapy.class_expression.OWLObjectIntersectionOf(
                        [top_ce._filler, new_property_value])
                    top_ce._filler = new_filler
                    return True
                elif isinstance(top_ce, owlapy.class_expression.OWLObjectUnionOf):
                    list_operands = list(top_ce.operands())
                    list_operands.remove(ce)
                    list_operands.append(
                        owlapy.class_expression.OWLObjectIntersectionOf(
                            [ce, new_property_value]))
                    top_ce._operands = tuple(list_operands)
                    return True
                raise ValueError("top_ce is not a valid value.")
        elif isinstance(ce, owlapy.class_expression.OWLNaryBooleanClassExpression):
            for i, op in enumerate(ce.operands()):
                if self.replace_nth_class(op, n, new_property_value, top_ce):
                    return True
                if isinstance(op, owlapy.class_expression.OWLClass):
                    n -= 1
        elif isinstance(ce, owlapy.class_expression.OWLObjectRestriction):
            return self.replace_nth_class(ce._filler, n, new_property_value, top_ce)

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

        eval_result['sparsity'] = np.mean(
            [explanation.sparsity for explanation in self.result.node_explanations])

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
        return CEBasedCore
