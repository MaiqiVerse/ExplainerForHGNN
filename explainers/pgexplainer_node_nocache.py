# > this version doesn't contain any cache related modifications
from .explainer import Explainer, ExplainerCore
import torch
import torch.nn as nn
import torch.nn.functional as F

from .explanation import NodeExplanation, NodeExplanationCombination
from .prepare_explanation_for_node_scores import standard_explanation
from .prepare_explanation_for_node_dataset_scores import prepare_explanation_fn_for_node_dataset_scores
from .prepare_combined_explanation_for_node_dataset_scores import prepare_combined_explanation_fn_for_node_dataset_scores
from .node_dataset_scores import node_dataset_scores


class PGExplainerNodeCore(ExplainerCore):
    """
    Core module for mask learning
    """
    def __init__(self, config):
        super().__init__(config)
        self.record_metrics = self.config.get("record_metrics", ["mask_density"])

    def init_params_node_level(self):
        """
        Define the initialization function for node-level explanation
        """
        gs, features = self.extract_neighbors_input()
        self.embedding_dim = features.shape[1]
        
        # > 2-layer MLP for the edges
        self.elayers = nn.Sequential(
            nn.Linear(self.embedding_dim * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(self.device)
        self.elayers.train()

        # > loss coefficients
        self.tmp = self.config.get("tmp", 0.1)
        self.sample_bias = self.config.get("sample_bias", 0.2)
        self.coeffs = {
            "size": self.config.get("coff_size", 0.01),
            "weight_decay": self.config.get("weight_decay", 0.005),
            "ent": self.config.get("coff_ent", 0.1),
            "connect": self.config.get("coff_connect", 0.0),
            "budget": self.config.get("budget", -1)
        }

    def concrete_sample(self, log_alpha, beta=1.0, training=True):
        """
        Concrete distribution for edge mask sampling
        """
        if training:
            bias = self.sample_bias
            noise = torch.rand_like(log_alpha) * (1.0 - 2 * bias) + bias
            gate_inputs = (torch.log(noise) - torch.log(1 - noise) + log_alpha) / beta
            return torch.sigmoid(gate_inputs)
        else:
            return torch.sigmoid(log_alpha)

    def forward_node_level(self):
        """
        Forward: mask edges -> model prediction
        """
        gs, features = self.extract_neighbors_input()
        
        # > organize data types
        target_dtype = self.elayers[0].weight.dtype
        g0 = gs[0].coalesce()
        edge_index = g0.indices().long()
        adj = g0.to_dense().to(target_dtype)
        features = features.to(target_dtype)

        # > calculate edge importance
        f1 = features[edge_index[0]]
        f2 = features[edge_index[1]]
        selfemb = features[self.mapping_node_id()].repeat(f1.shape[0], 1)
        h = torch.cat([f1, f2, selfemb], dim=-1)
        values = self.elayers(h).squeeze()
        sampled = self.concrete_sample(values, beta=self.tmp, training=True).to(target_dtype)

        # > apply symmetrized masks
        mask = torch.zeros_like(adj)
        mask[edge_index[0], edge_index[1]] = sampled
        mask = (mask + mask.T) / 2
        masked_adj = adj * mask
        masked_adj.fill_diagonal_(0)

        # > convert back to sparse
        try:
            masked_sparse = masked_adj.to_sparse()
        except AttributeError:
            masked_sparse = masked_adj.to_sparse_coo()

        self.masked = {
            "masked_gs": [masked_sparse],
            "masked_features": features,
        }

        def handle_fn(model):
            return [masked_sparse], features

        output = self.model.custom_forward(handle_fn)

        self.mask = mask
        self.masked_adj = masked_adj
        self.pred = F.softmax(output[self.mapping_node_id()], dim=0)
        self.adj_tensor = adj
        
        return self.get_loss(self.pred)

    def get_loss(self, output, mask=None):
        return self.get_loss_node_level(output, mask)

    def get_loss_node_level(self, pred, mask=None):
        """
        Loss components
        """
        label = self._target_class_for_node(self.node_id)
        assert 0 <= label < pred.shape[0]
        pred_loss = -torch.log(pred[label].clamp_min(1e-6))

        # Size loss
        if self.coeffs["budget"] <= 0:
            size_loss = self.coeffs["size"] * torch.sum(self.mask)
        else:
            size_loss = self.coeffs["size"] * torch.relu(torch.sum(self.mask) - self.coeffs["budget"])

        # Entropy loss
        scale = 0.99
        mask = self.mask * (2 * scale - 1.0) + (1.0 - scale)
        ent_loss = -mask * torch.log(mask + 1e-6) - (1 - mask) * torch.log(1 - mask + 1e-6)
        mask_ent_loss = self.coeffs["ent"] * torch.mean(ent_loss)

        # L2 regularization
        l2_loss = sum(torch.norm(p) for p in self.elayers.parameters() if p.requires_grad)
        l2_loss *= self.coeffs["weight_decay"]
        
        loss = pred_loss + size_loss + l2_loss + mask_ent_loss

        # Connectivity loss (optional)
        connect_loss = 0.0
        if self.coeffs["budget"] > 0 and self.coeffs["connect"] > 0:
            adj_tensor_dense = self.adj_tensor
            noise = torch.rand_like(adj_tensor_dense) * 0.001
            adj_tensor_dense = adj_tensor_dense + noise

            cols = torch.argsort(adj_tensor_dense, dim=-1, descending=True)
            sampled_rows = torch.arange(adj_tensor_dense.shape[0]).unsqueeze(1)
            sampled_cols_0 = cols[:, 0].unsqueeze(1)
            sampled_cols_1 = cols[:, 1].unsqueeze(1)

            sampled0 = torch.cat([sampled_rows, sampled_cols_0], dim=-1)
            sampled1 = torch.cat([sampled_rows, sampled_cols_1], dim=-1)

            sample0_score = self.mask[sampled0[:, 0], sampled0[:, 1]]
            sample1_score = self.mask[sampled1[:, 0], sampled1[:, 1]]

            connect_loss = -((1.0 - sample0_score) * torch.log(1.0 - sample1_score + 1e-6) +
                             sample0_score * torch.log(sample1_score + 1e-6))
            connect_loss = self.coeffs["connect"] * torch.sum(connect_loss)
        
        loss = loss + connect_loss

        # > save all the losses
        self._last_pred_loss = pred_loss.detach()
        self._last_size_loss = size_loss.detach()
        self._last_ent_loss = mask_ent_loss.detach()
        self._last_l2_loss = l2_loss.detach()
        self._last_connect_loss = (connect_loss.detach() if isinstance(connect_loss, torch.Tensor) 
                                   else torch.tensor(connect_loss))

        return loss

    def fit_node_level(self):
        """
        Define the training loop for the explainer model
        """
        self.model.eval()
        self._label_split_idx = 0  # > use train labels
        
        # > get train nodes
        train_nodes = self.config.get("train_node_ids", None)
        if train_nodes is None:
            train_nodes = [node for node, _ in self.model.dataset.labels[0]]
        
        # > initialize the parameters with the 1st node
        self.node_id = train_nodes[0]
        self.init_params_node_level()
        
        optimizer = torch.optim.Adam(self.elayers.parameters(), lr=self.config.get("opt_lr", 0.01))
        
        # > train loop
        for epoch in range(self.config.get("epochs", 30)):
            total_loss = 0.0
            
            for node_id in train_nodes:
                self.node_id = node_id
                # > Clear cache and reconstruct subgraph for each iteration
                if hasattr(self, "neighbor_input"):
                    self.neighbor_input = {}
                for attr in ("used_nodes", "recovery_dict", "_quick_transfer", "mapped_node_id"):
                    if hasattr(self, attr):
                        delattr(self, attr)
                
                loss = self.forward_node_level()
                total_loss = total_loss + loss
            
            total_loss = total_loss / len(train_nodes)
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            self.current_loss = float(total_loss.detach().cpu())

    def explain(self, model, node_id):
        """
        After training, produce a NodeExplanation for a single node
        """
        self.model = model
        self.model.eval()
        self.node_id = node_id
        self._label_split_idx = 2  # > test labels

        # > clear cache
        if hasattr(self, "neighbor_input"):
            self.neighbor_input = {}
        for attr in ("used_nodes", "recovery_dict", "_quick_transfer", "mapped_node_id"):
            if hasattr(self, attr):
                delattr(self, attr)

        if not hasattr(self, "elayers"):
            self.init_params_node_level()

        _ = self.forward_node_level()

        explanation = NodeExplanation()
        explanation = standard_explanation(explanation, self)
        return explanation

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
        Extract n_hop neighborhood subgraph centered on self.node_id
        """
        if not self.config.get('extract_neighbors', True):
            gs, features = self.model.standard_input()
            self.neighbor_input = {"gs": gs, "features": features}
            return gs, features

        if getattr(self, 'neighbor_input', None) is not None and self.neighbor_input.get("gs", None) is not None:
            return self.neighbor_input["gs"], self.neighbor_input["features"]

        self.n_hop = self.config.get('n_hop', 2)
        gs, features = self.model.standard_input()

        used_nodes_set = set()
        for g in gs:
            indices = g.indices()
            current_nodes = [self.node_id]
            for _ in range(self.n_hop):
                new_current_nodes = set()
                for node in current_nodes:
                    mask = (indices[0] == node) | (indices[1] == node)
                    used_nodes_set.update(indices[0][mask].tolist())
                    used_nodes_set.update(indices[1][mask].tolist())
                    new_current_nodes.update(indices[0][mask].tolist())
                    new_current_nodes.update(indices[1][mask].tolist())
                current_nodes = list(new_current_nodes)

        self.used_nodes = sorted(list(used_nodes_set))
        self.recovery_dict = {node: i for i, node in enumerate(self.used_nodes)}
        
        self._quick_transfer = torch.zeros(len(features), dtype=torch.long).to(self.device_string)
        for i, node in enumerate(self.used_nodes):
            self._quick_transfer[node] = i

        temp_used_nodes_tensor = torch.tensor(self.used_nodes).to(self.device_string)
        new_gs = []
        for g in gs:
            indices = g.indices()
            mask = torch.isin(indices[0], temp_used_nodes_tensor) & torch.isin(indices[1], temp_used_nodes_tensor)
            new_indices = torch.stack(
                [self._quick_transfer[indices[0][mask]], self._quick_transfer[indices[1][mask]]],
                dim=0
            ).to(self.device_string)
            new_values = g.values()[mask]
            shape = torch.Size([len(self.used_nodes), len(self.used_nodes)])
            new_gs.append(torch.sparse_coo_tensor(new_indices, new_values, shape))

        sub_features = features[self.used_nodes]
        self.neighbor_input = {"gs": new_gs, "features": sub_features}
        return new_gs, sub_features

    def _label_from_split(self, nid, split_idx):
        return dict(self.model.dataset.labels[split_idx])[nid]

    def _get_label_lookup(self):
        if not hasattr(self, "_label_lookup"):
            split_idx = getattr(self, "_label_split_idx", 2)
            self._label_lookup = dict(self.model.dataset.labels[split_idx])
        return self._label_lookup

    def _get_pred_label_lookup(self):
        if not hasattr(self, "_pred_label_lookup"):
            gs, features = self.model.standard_input()
            with torch.no_grad():
                logits = self.model.custom_forward(lambda m: (gs, features))
            self._pred_label_lookup = logits.argmax(dim=1).detach().cpu().tolist()
        return self._pred_label_lookup

    def _target_class_for_node(self, global_node_id, pred_on_masked=None):
        if self.config.get("use_pred_label", False):
            return int(self._get_pred_label_lookup()[int(global_node_id)])
        else:
            lookup = self._get_label_lookup()
            return int(lookup[int(global_node_id)])

    def get_required_fit_params(self):
        return list(self.elayers.parameters())

    def get_input_handle_fn_node_level(self):
        def handle_fn(model):
            return self.extract_neighbors_input()
        return handle_fn


class PGExplainerMeta(Explainer):
    def __init__(self, config):
        super().__init__(config)

    def explain(self, model, **kwargs):
        self.model = model
        if self.model.dataset.single_graph:
            return self.node_level_explain(**kwargs)
        else:
            return self.graph_level_explain(**kwargs)

    def node_level_explain(self, **kwargs):
        """
        Run PGExplainerCore's training once over multiple nodes, then explain each node
        """
        explainer_core = PGExplainerNodeCore(self.config)
        explainer_core.to(self.device)
        explainer_core.model = self.model

        # > train once
        explainer_core.fit_node_level()
        
        # > explain test node
        result = []
        test_labels = self.model.dataset.labels[2]
        if kwargs.get("max_nodes"):
            test_labels = test_labels[:kwargs["max_nodes"]]
        
        for idx, _ in test_labels:
            explanation = explainer_core.explain(self.model, node_id=idx)
            result.append(explanation)
        
        # > evaluate results
        self.result = self.construct_explanation(result)
        self.evaluate()
        self.save_summary()
        return self.eval_result

    def construct_explanation(self, result):
        result = NodeExplanationCombination(node_explanations=result)
        if self.config.get("control_data"):
            result.control_data = self.config["control_data"]
        return result

    def evaluate(self):
        eval_result = {}
        if self.config.get("eval_metrics"):
            for metric in self.config["eval_metrics"]:
                self.result = prepare_combined_explanation_fn_for_node_dataset_scores[metric](self.result, self)
                eval_result[metric] = node_dataset_scores[metric](self.result)
        self.eval_result = eval_result
        return eval_result
