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
    Core module for mask learning with optimized subgraph caching
    """
    def __init__(self, config):
        super().__init__(config)
        self.record_metrics = self.config.get("record_metrics", ["mask_density"])
        # > initialize subgraph cache
        self.subgraph_cache = {}
        #self.device_string = 'cuda' if torch.cuda.is_available() else 'cpu' # AttributeError: can't set attribute 'device_string'

    def init_params_node_level(self, train_nodes=None):
        """
        Initialize explainer MLP and cache all training subgraphs
        """
        # > if the train node list is provided, cache all subgraphs in advance
        if train_nodes is not None:
            self._cache_all_subgraphs(train_nodes)
        
        # > get the feature dims with the 1st node
        if train_nodes:
            self.node_id = train_nodes[0]
        gs, features = self.extract_neighbors_input()

        # > get the actual embedding dimension from the model
        with torch.no_grad():
            embeddings = self.model.embedding(lambda m: (gs, features))
            self.embedding_dim = embeddings.shape[1]  # > use embedding dimension not feature dimension
            #self.embedding_dim = features.shape[1]
        print(f"Embedding dimension: {self.embedding_dim}")
        
        # > define a 2-layer MLP to predict the importance of each edge
        self.elayers = nn.Sequential(
            nn.Linear(self.embedding_dim * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(self.device)

        # > better MLP initialization
        for layer in self.elayers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.1)
        #        if layer.out_features == 1:
        #            # > initialize final layer to output positive values
        #            nn.init.constant_(layer.bias, 1.0)
        #        else:
                nn.init.zeros_(layer.bias)

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

    def _cache_all_subgraphs(self, node_list):
        """
        Cache all caches of all nodes to avoid repetitive computation
        """
        print(f"Caching subgraphs for {len(node_list)} nodes...")

        # > Pre-compute all embeddings ONCE
        with torch.no_grad():
            for i, node_id in enumerate(node_list):
                if i % 20 == 0:
                    print(f"Cached {i}/{len(node_list)} nodes...")
            #for node_id in node_list:
                if node_id not in self.subgraph_cache:
                    self.node_id = node_id
                    # > clear the previously cached 
                    self._clear_node_cache()
                    # > construct and cache the subgraph
                    gs, features = self._extract_and_cache_neighbors(node_id)

                    # > pre-compute embeddings for this subgraph
                    embeddings = self.model.embedding(lambda m: (gs, features))
                    embeddings = embeddings.detach().cpu()  # store as CPU tensor

                    # > cache the infos
                    self.subgraph_cache[node_id] = {
                        'gs': gs,
                        'features': features,
                        'embeddings': embeddings,
                        'mapped_node_id': self.mapping_node_id(),
                        'used_nodes': getattr(self, 'used_nodes', None),
                        'recovery_dict': getattr(self, 'recovery_dict', None),
                        '_quick_transfer': getattr(self, '_quick_transfer', None)
                    }
        print(f"Subgraph caching completed!")

    def _clear_node_cache(self):
        """
        Clear all the relevant cache of this node
        """
        self.neighbor_input = {}
        for attr in ("used_nodes", "recovery_dict", "_quick_transfer", "mapped_node_id"):
            if hasattr(self, attr):
                delattr(self, attr)

    def _extract_and_cache_neighbors(self, node_id):
        """
        Extract and cache the neighbor subgraphs
        """
        # temp node_id
        old_node_id = getattr(self, 'node_id', None)
        self.node_id = node_id
        
        # > do not use extracted neighbors, return the full graph
        if not self.config.get('extract_neighbors', True):
            gs, features = self.model.standard_input()
            self.neighbor_input = {"gs": gs, "features": features}
            self.node_id = old_node_id
            return gs, features

        # > Extract n-hop neighbors
        self.n_hop = self.config.get('n_hop', 2)
        gs, features = self.model.standard_input()

        used_nodes_set = set()
        for g in gs:
            indices = g.indices()
            current_nodes = [node_id]
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
        
        # > construct quick transfer tensor
        #self._quick_transfer = torch.zeros(len(features), dtype=torch.long).to(self.device_string) # AttributeError: can't set attribute 'device_string'
        self._quick_transfer = torch.zeros(len(features), dtype=torch.long).to(self.device)
        for i, node in enumerate(self.used_nodes):
            self._quick_transfer[node] = i

        # > reconstruct subgraph sparse adj matrix
        #temp_used_nodes_tensor = torch.tensor(self.used_nodes).to(self.device_string) # AttributeError: can't set attribute 'device_string'
        temp_used_nodes_tensor = torch.tensor(self.used_nodes).to(self.device)
        new_gs = []
        for g in gs:
            indices = g.indices()
            mask = torch.isin(indices[0], temp_used_nodes_tensor) & torch.isin(indices[1], temp_used_nodes_tensor)
            new_indices = torch.stack(
                [self._quick_transfer[indices[0][mask]], self._quick_transfer[indices[1][mask]]],
                dim=0
            #).to(self.device_string)  # AttributeError: can't set attribute 'device_string'
            ).to(self.device)
            new_values = g.values()[mask]
            shape = torch.Size([len(self.used_nodes), len(self.used_nodes)])
            new_gs.append(torch.sparse_coo_tensor(new_indices, new_values, shape))

        sub_features = features[self.used_nodes]
        self.neighbor_input = {"gs": new_gs, "features": sub_features}
        
        self.node_id = old_node_id
        return new_gs, sub_features

    def extract_neighbors_input(self):
        """
        Extract neighbor subgraphs from cache
        """
        # > check cache first
        if self.node_id in self.subgraph_cache:
            cached = self.subgraph_cache[self.node_id]
            # > restore the attributes of the cache
            self.used_nodes = cached['used_nodes']
            self.recovery_dict = cached['recovery_dict']
            self._quick_transfer = cached['_quick_transfer']
            self.mapped_node_id = cached['mapped_node_id']
            self.neighbor_input = {"gs": cached['gs'], "features": cached['features']}
            return cached['gs'], cached['features']
        
        # > if no cache, use the original method
        return self._extract_and_cache_neighbors(self.node_id)

    def concrete_sample(self, log_alpha, beta=1.0, training=True):
        """Concrete distribution for edge mask sampling"""
        if training:
            bias = self.sample_bias
            #noise = torch.rand_like(log_alpha) * (1.0 - 2 * bias) + bias
            random_noise = torch.rand_like(log_alpha) * (1.0 - 2 * bias) + bias
            random_noise = random_noise.clamp(min=1e-7, max=1-1e-7)  # to avoid log(0)
            #gate_inputs = (torch.log(noise) - torch.log(1 - noise) + log_alpha) / beta
            gate_inputs = torch.log(random_noise) - torch.log(1.0 - random_noise)
            gate_inputs = (gate_inputs + log_alpha) / beta
            gate_inputs = torch.sigmoid(gate_inputs)
            #return torch.sigmoid(gate_inputs)
        else:
            gate_inputs = torch.sigmoid(log_alpha)
            #return torch.sigmoid(log_alpha)
        return gate_inputs

    def forward_node_level(self):
        """Forward pass for a single node"""
        #gs, features = self.extract_neighbors_input()

        # > get MLP embeddings
        #with torch.no_grad():
        #    embeddings = self.model.embedding(lambda m: (gs, features))
        #    embeddings = embeddings.detach()
        #    embeddings = F.normalize(embeddings, p=2, dim=1)  # L2 normalize

        # Check if embeddings are cached
        if self.node_id in self.subgraph_cache and 'embeddings' in self.subgraph_cache[self.node_id]:
            cached = self.subgraph_cache[self.node_id]
            gs = cached['gs']
            features = cached['features']
            embeddings = cached['embeddings'].to(self.device)  # Use pre-computed embeddings
        else:
            # Fall back to computing embeddings on the fly
            gs, features = self.extract_neighbors_input()
            with torch.no_grad():
                embeddings = self.model.embedding(lambda m: (gs, features))
                embeddings = embeddings.detach()
        
        # > organize the data type
        #target_dtype = self.elayers[0].weight.dtype
        target_dtype = torch.float32
        g0 = gs[0].coalesce()
        edge_index = g0.indices().long()
        adj = g0.to_dense().to(target_dtype)
        #features = features.to(target_dtype)
        embeddings = embeddings.to(target_dtype)

        # > calculate edge importances
        #f1 = features[edge_index[0]]
        #f2 = features[edge_index[1]]
        f1 = embeddings[edge_index[0]]
        f2 = embeddings[edge_index[1]]
        #selfemb = features[self.mapping_node_id()].repeat(f1.shape[0], 1)
        #selfemb = features[self.mapping_node_id()].unsqueeze(0).repeat(f1.shape[0], 1)
        selfemb = embeddings[self.mapping_node_id()].unsqueeze(0).repeat(f1.shape[0], 1)
        #h = torch.cat([f1, f2, selfemb], dim=-1)
        f12self = torch.cat([f1, f2, selfemb], dim=-1)

        # > MLP
        h = self.elayers(f12self)
        #values = self.elayers(h).squeeze()
        values = h.squeeze(-1)

        #values = values * 0.1  # > scale down the logits
        values = torch.clamp(values, min=-5, max=5)

        if self.node_id == self.debug_node and hasattr(self, 'current_epoch'):
            print(f"Epoch {self.current_epoch}: Raw values min={values.min():.4f}, max={values.max():.4f}, mean={values.mean():.4f}")
        # Use temperature starting high and decreases
        temperature = 1.0 if not hasattr(self, 'current_epoch') else max(0.1, 1.0 - self.current_epoch * 0.02)

        #sampled = self.concrete_sample(values, beta=self.tmp, training=True).to(target_dtype)
        #sampled = self.concrete_sample(values, beta=self.tmp, training=True)
        sampled = self.concrete_sample(values, beta=temperature, training=True)

        if self.node_id == self.debug_node and hasattr(self, 'current_epoch'):
            print(f"  Sampled mask min={sampled.min():.4f}, max={sampled.max():.4f}, mean={sampled.mean():.4f}")

        # > use symetrized edge mask
        mask = torch.zeros_like(adj)
        mask[edge_index[0], edge_index[1]] = sampled
        mask = (mask + mask.T) / 2
        masked_adj = adj * mask
        masked_adj.fill_diagonal_(0)

        # > convert back to sparse
        masked_sparse = masked_adj.to_sparse_coo()

        output = self.model.custom_forward(lambda m: ([masked_sparse], features))
        node_output = output[self.mapping_node_id()]
        pred = F.softmax(node_output, dim=0)

        self.mask = mask
        self.masked_adj = masked_adj
        self.pred = pred
        self.adj_tensor = adj
        
        return self.get_loss(self.pred)

    def get_loss(self, output, mask=None):
        return self.get_loss_node_level(output, mask)

    def get_loss_node_level(self, pred, mask=None):
        """
        Calculate loss
        """
        if self.config.get("use_pred_label", False):
            # > get original predictions
            gs, features = self.extract_neighbors_input()
            with torch.no_grad():
                orig_output = self.model.custom_forward(lambda m: (gs, features))
                pred_label = orig_output[self.mapping_node_id()].argmax().item()
            target_class = pred_label
        else:
            target_class = self._target_class_for_node(self.node_id)

        #label = self._target_class_for_node(self.node_id)
        #assert 0 <= label < pred.shape[0]

        logit = pred[target_class]
        logit = logit.clamp(min=1e-6)  # to avoid log(0)
        #pred_loss = -torch.log(pred[label].clamp_min(1e-6))
        pred_loss = -torch.log(logit)

        # Size loss
        if self.coeffs["budget"] <= 0:
            size_loss = self.coeffs["size"] * torch.sum(self.mask)
        else:
            size_loss = self.coeffs["size"] * torch.relu(torch.sum(self.mask) - self.coeffs["budget"])

        # Entropy loss
        scale = 0.99
        #mask = self.mask * (2 * scale - 1.0) + (1.0 - scale)
        mask_for_ent = self.mask * (2 * scale - 1.0) + (1.0 - scale)
        mask_for_ent = mask_for_ent.clamp(min=1e-7, max=1-1e-7)
        #ent_loss = -mask * torch.log(mask + 1e-6) - (1 - mask) * torch.log(1 - mask + 1e-6)
        mask_ent = -mask_for_ent * torch.log(mask_for_ent) - (1 - mask_for_ent) * torch.log(1 - mask_for_ent)
        #mask_ent_loss = self.coeffs["ent"] * torch.mean(ent_loss)
        mask_ent_loss = self.coeffs["ent"] * torch.mean(mask_ent)


        # L2 regularization
        l2_loss = 0
        #l2_loss = sum(torch.norm(p) for p in self.elayers.parameters() if p.requires_grad)
        for param in self.elayers.parameters():
            if len(param.shape) > 1:
                l2_loss += torch.norm(param)
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

        # > save the losses for further use
        self._last_pred_loss = pred_loss.detach()
        self._last_size_loss = size_loss.detach()
        self._last_ent_loss = mask_ent_loss.detach()
        self._last_l2_loss = l2_loss.detach()
        self._last_connect_loss = (connect_loss.detach() if isinstance(connect_loss, torch.Tensor) 
                                   else torch.tensor(connect_loss))

        return loss

    def fit_node_level(self):
        """
        Training loop with per-node updates to avoid gradient explosion
        """
        self.model.eval()
        self._label_split_idx = 0  # use training labels
    
        # Get the training nodes
        train_nodes = self.config.get("train_node_ids", None)
        if train_nodes is None:
            train_nodes = [node for node, _ in self.model.dataset.labels[0]]
        # To avoid OOM for now
        train_nodes = train_nodes[:200]
    
        # Initialize parameters and cache all the subgraphs
        self.init_params_node_level(train_nodes)
    
        # Create optimizer ONCE
        optimizer = torch.optim.Adam(self.elayers.parameters(), lr=self.config.get("opt_lr", 0.01))
    
        # Track first node for debugging
        self.debug_node = train_nodes[0]
    
        # Training loop
        for epoch in range(self.config.get("epochs", 50)):
            self.current_epoch = epoch
            epoch_loss = 0.0
        
            # Update per node (not accumulating gradients)
            for node_id in train_nodes:
                self.node_id = node_id
            
                # Clear gradients for each node
                optimizer.zero_grad()
            
                # Forward pass for single node
                loss = self.forward_node_level()
            
                # Backward pass for single node
                loss.backward()
            
                # Gradient clipping to prevent explosion
                torch.nn.utils.clip_grad_norm_(self.elayers.parameters(), max_norm=1.0)
            
                # Update weights immediately
                optimizer.step()
            
                # Track loss for logging
                epoch_loss += loss.item()
        
            # Calculate average loss for the epoch
            avg_loss = epoch_loss / len(train_nodes)
            self.current_loss = avg_loss
        
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Avg Loss = {avg_loss:.4f}")

    def explain(self, model, node_id):
        """
        Get the explanation of one node
        """
        self.model = model
        self.model.eval()
        self.node_id = node_id
        self._label_split_idx = 2  # use test labels

        # > clear cache
        if hasattr(self, "neighbor_input"):
            self.neighbor_input = {}
        for attr in ("used_nodes", "recovery_dict", "_quick_transfer", "mapped_node_id"):
            if hasattr(self, attr):
                delattr(self, attr)

        if not hasattr(self, "elayers"):
            self.init_params_node_level()

        _ = self.forward_node_level() # > generate self.mask (soft mask)

        # > the original soft mask
        soft_mask = self.mask.clone()

        # > convert to hard mask with sparsity of 25%
        target_sparsity = self.config.get("top_k_for_edge_mask", 0.25)

        # > get upper triangular indices (to avoid duplicate counting in undirected graphs)
        n = soft_mask.shape[0]
        #triu_indices = torch.triu_indices(n, n, offset=1)
        triu_indices = torch.triu_indices(n, n, offset=1, device=soft_mask.device)
        edge_values = soft_mask[triu_indices[0], triu_indices[1]]

        # > calculate how many edges to keep
        num_edges = edge_values.numel()

        # > Handle edge cases
        if num_edges == 0:
            # > No edges in the subgraph
            print(f"Node {node_id}: No edges in subgraph")
            hard_mask = torch.zeros_like(soft_mask)
        else:
            k = int(num_edges * target_sparsity)
            #k = max(1, k)
            k = max(1, min(k, num_edges))  # Ensure k is between 1 and num_edges

            # > Check if all edge values are zero (mask not learned)
            if edge_values.max() == 0:
                # > If soft mask is all zeros, select k random edges
                print(f"Node {node_id}: Soft mask is all zeros, selecting random edges")
                random_indices = torch.randperm(num_edges, device=soft_mask.device)[:k]
                hard_mask = torch.zeros_like(soft_mask)
                selected_edges = triu_indices[:, random_indices]
                hard_mask[selected_edges[0], selected_edges[1]] = 1.0
                hard_mask[selected_edges[1], selected_edges[0]] = 1.0
            else:
                # Normal case: select top-k edges

                # > get top k values
                topk_values, topk_indices = torch.topk(edge_values, k)

                # > build hard mask
                hard_mask = torch.zeros_like(soft_mask)
                selected_edges = triu_indices[:, topk_indices]
                hard_mask[selected_edges[0], selected_edges[1]] = 1.0
                hard_mask[selected_edges[1], selected_edges[0]] = 1.0  # 对称

        # > replace self.mask with hard mask
        #self.mask = hard_mask  # standard_explanation
        #self.soft_mask_backup = soft_mask  # backup soft mask

            actual_sparsity = k / num_edges if num_edges > 0 else 0
            print(f"Node {node_id}: Soft mask mean: {soft_mask.mean():.4f}, "
                f"Hard mask sparsity: {actual_sparsity:.4f} (target: {target_sparsity})")

        # > Replace self.mask with hard mask
        self.mask = hard_mask
        self.soft_mask_backup = soft_mask

        explanation = NodeExplanation()
        explanation = standard_explanation(explanation, self)

        # > attributes for fidelity
        with torch.no_grad():
            # > get original predictions
            gs, features = self.extract_neighbors_input()
            output_orig = self.model.custom_forward(lambda m: (gs, features))

            # > get masked predictions with hard mask
            g0 = gs[0].coalesce()
            adj = g0.to_dense()
            # > Apply HARD mask for masked predictions
            masked_adj = adj * self.mask  # Now using hard mask
            masked_adj.fill_diagonal_(0)

            try:
                masked_sparse = masked_adj.to_sparse()
            except AttributeError:
                masked_sparse = masked_adj.to_sparse_coo()
            #output_masked = self.model.custom_forward(lambda m: (self.masked["masked_gs"],
                                                            #self.masked["masked_features"]))
            output_masked = self.model.custom_forward(lambda m: ([masked_sparse], features))

            # > get opposite masked predictions (complement of the mask)
            # Get the original adjacency and create opposite mask
            #g0 = gs[0].coalesce()
            #adj = g0.to_dense()
            opposite_mask = 1.0 - self.mask  # Complement of the mask
            opposite_masked_adj = adj * opposite_mask
            opposite_masked_adj.fill_diagonal_(0)

            # Convert to sparse
            try:
                opposite_masked_sparse = opposite_masked_adj.to_sparse()
            except AttributeError:
                opposite_masked_sparse = opposite_masked_adj.to_sparse_coo()

            # Get predictions with opposite mask
            output_opposite_masked = self.model.custom_forward(
                lambda m: ([opposite_masked_sparse], features)
            )

            # > attributes
            explanation.node_id = node_id
            # Convert label to tensor
            explanation.label = torch.tensor(self._target_class_for_node(node_id), dtype=torch.long)
            # Get the mapped node index for subgraph
            mapped_idx = self.mapping_node_id()
            # original pred
            #explanation.pred = output_orig[self.mapping_node_id()]
            explanation.pred = output_orig[mapped_idx]
            # Keep as tensor, not .item()
            #explanation.pred_label_hard = output_orig[self.mapping_node_id()].argmax().item()
            #explanation.pred_label_hard = output_orig[self.mapping_node_id()].argmax(dim=0, keepdim=True)
            explanation.pred_label_hard = output_orig[mapped_idx].argmax(dim=0, keepdim=True)
            # masked pred (with explanation mask)
            #explanation.masked_pred = output_masked[self.mapping_node_id()]
            explanation.masked_pred = output_masked[mapped_idx]
            #explanation.masked_pred_label_hard = output_masked[self.mapping_node_id()].argmax().item()
            #explanation.masked_pred_label_hard = output_masked[self.mapping_node_id()].argmax(dim=0, keepdim=True)
            explanation.masked_pred_label_hard = output_masked[mapped_idx].argmax(dim=0, keepdim=True)
            explanation.masked_pred_label = output_masked[mapped_idx]

            # opposite masked pred (with complement of explanation mask)
            explanation.opposite_masked_pred = output_opposite_masked[mapped_idx]
            explanation.opposite_masked_pred_label_hard = output_opposite_masked[mapped_idx].argmax(dim=0, keepdim=True)

        return explanation

    def mapping_node_id(self):
        """Project the node id to the subgraph"""
        if getattr(self, 'mapped_node_id', None) is not None:
            return self.mapped_node_id
        if not self.config.get('extract_neighbors', True):
            self.mapped_node_id = self.node_id
        else:
            self.mapped_node_id = self.recovery_dict[self.node_id]
        return self.mapped_node_id

    def _label_from_split(self, nid, split_idx):
        """Get node label from the specified data split"""
        return dict(self.model.dataset.labels[split_idx])[nid]

    def _get_label_lookup(self):
        """Get the true labels"""
        if not hasattr(self, "_label_lookup"):
            split_idx = getattr(self, "_label_split_idx", 2)
            self._label_lookup = dict(self.model.dataset.labels[split_idx])
        return self._label_lookup

    def _get_pred_label_lookup(self):
        """Get the predicted labels"""
        if not hasattr(self, "_pred_label_lookup"):
            gs, features = self.model.standard_input()
            with torch.no_grad():
                logits = self.model.custom_forward(lambda m: (gs, features))
            self._pred_label_lookup = logits.argmax(dim=1).detach().cpu().tolist()
        return self._pred_label_lookup

    def _target_class_for_node(self, global_node_id, pred_on_masked=None):
        """Get the label of the target node"""
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
        """Node level explanation. Explain multiple nodes by just training once"""
        # > core explainer
        explainer_core = PGExplainerNodeCore(self.config)
        explainer_core.to(self.device)
        explainer_core.model = self.model

        # > 
        explainer_core.fit_node_level()
        
        # > explain the test nodes
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
        """Combine all the explanation of all the nodes"""
        result = NodeExplanationCombination(node_explanations=result)
        if self.config.get("control_data"):
            result.control_data = self.config["control_data"]
        return result

    def evaluate(self):
        """Evaluate the quality of the mask"""
        eval_result = {}
        if self.config.get("eval_metrics"):
            for metric in self.config["eval_metrics"]:
                self.result = prepare_combined_explanation_fn_for_node_dataset_scores[metric](self.result, self)
                eval_result[metric] = node_dataset_scores[metric](self.result)
        self.eval_result = eval_result
        return eval_result
