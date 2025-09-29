from .model import BaseModel
from datasets.dataset import NodeClassificationDataset
from datasets import ACM, DBLP, IMDB
from .utils import node_classification_support_metrics

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import json


class GATConv(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads, dropout=0.0, attn_dropout=0.0,
                 negative_slope=0.2, residual=False, activation=None,
                 allow_zero_in_degree=False, bias=True, *args, **kwargs
                 ):
        r"""
        Graph attention layer from `Graph Attention Network
        <https://arxiv.org/pdf/1710.10903.pdf>`__

        .. math::
            h_i^{(l+1)} = \sum_{j\in \mathcal{N}(i)} \alpha_{ij} W^{(l)} h_j^{(l)}

        where :math:`\alpha_{ij}` is the attention score bewteen node :math:`i` and
        node :math:`j`:

        .. math::
            \alpha_{ij}^{l} &= \mathrm{softmax_i} (e_{ij}^{l})

            e_{ij}^{l} &= \mathrm{LeakyReLU}\left(\vec{a}^T [W h_{i} \| W h_{j}]\right)

        Parameters
        ----------
        in_feats : int, or pair of ints
            Input feature size; i.e, the number of dimensions of :math:`h_i^{(l)}`.
            GATConv can be applied on homogeneous graph and unidirectional
            `bipartite graph.
            If the layer is to be applied to a unidirectional bipartite graph, ``in_feats``
            specifies the input feature size on both the source and destination nodes.  If
            a scalar is given, the source and destination node feature size would take the
            same value.
        out_feats : int
            Output feature size; i.e, the number of dimensions of :math:`h_i^{(l+1)}`.
        num_heads : int
            Number of heads in Multi-Head Attention.
        feat_drop : float, optional
            Dropout rate on feature. Defaults: ``0``.
        attn_drop : float, optional
            Dropout rate on attention weight. Defaults: ``0``.
        negative_slope : float, optional
            LeakyReLU angle of negative slope. Defaults: ``0.2``.
        residual : bool, optional
            If True, use residual connection. Defaults: ``False``.
        activation : callable activation function/layer or None, optional.
            If not None, applies an activation function to the updated node features.
            Default: ``None``.
        allow_zero_in_degree : bool, optional
            If there are 0-in-degree nodes in the graph, output for those nodes will be invalid
            since no message will be passed to those nodes. This is harmful for some applications
            causing silent performance regression. This module will raise a DGLError if it detects
            0-in-degree nodes in input graph. By setting ``True``, it will suppress the check
            and let the users handle it by themselves. Defaults: ``False``.
        bias : bool, optional
            If True, learns a bias term. Defaults: ``True``.

        Note
        ----
        Zero in-degree nodes will lead to invalid output value. This is because no message
        will be passed to those nodes, the aggregation function will be appied on empty input.
        A common practice to avoid this is to add a self-loop for each node in the graph if
        it is homogeneous, which can be achieved by:

        >>> g = ... # a scipy sparse matrix
        >>> g = g + sp.eye(g.shape[0])
        >>> g = g.eliminate_zeros()
        >>> indices = torch.from_numpy(np.asarray([g.row, g.col]))
        >>> values = torch.from_numpy(g.data)
        >>> g = torch.sparse_coo_tensor(indices, values, g.shape)

        Examples
        --------
        >>> import numpy as np
        >>> import torch
        >>> import scipy.sparse as sp
        >>> row_indices = np.array([0,1,2,3,2,5])
        >>> col_indices = np.array([1,2,3,4,0,3])
        >>> values = np.ones(6)
        >>> g = sp.coo_matrix((values, (row_indices, col_indices)))
        >>> g = g + sp.eye(g.shape[0])
        >>> g = g.eliminate_zeros()
        >>> indices = torch.from_numpy(np.asarray([g.row, g.col]))
        >>> values = torch.from_numpy(g.data)
        >>> g = torch.sparse_coo_tensor(indices, values, g.shape)
        >>> feat = torch.ones(6, 10)
        >>> gatconv = GATLayer(10, 2, 3)
        >>> res = gatconv(feat, g)
        >>> res
        tensor([[[-0.6066,  1.0268],
                [-0.5945, -0.4801],
                [ 0.1594,  0.3825]],
                [[ 0.0268,  1.0783],
                [ 0.5041, -1.3025],
                [ 0.6568,  0.7048]],
                [[-0.2688,  1.0543],
                [-0.0315, -0.9016],
                [ 0.3943,  0.5347]],
                [[-0.6066,  1.0268],
                [-0.5945, -0.4801],
                [ 0.1594,  0.3825]]], grad_fn=<BinaryReduceBackward>)

        >>> row_indices = np.array([0, 1, 0, 0, 1])
        >>> col_indices = np.array([0, 1, 2, 3, 2])
        >>> values = np.ones(5)
        >>> g = sp.coo_matrix((values, (row_indices, col_indices)))
        >>> g = torch.sparse_coo_tensor(torch.from_numpy(np.asarray([g.row, g.col])), torch.from_numpy(g.data), g.shape)
        >>> u_feat = torch.tensor(np.random.rand(2, 5).astype(np.float32))
        >>> v_feat = torch.tensor(np.random.rand(4, 10).astype(np.float32))
        >>> gatconv = GATLayer((5,10), 2, 3)
        >>> res = gatconv((u_feat, v_feat), g)
        >>> res
        tensor([[[-0.6066,  1.0268],
                [-0.5945, -0.4801],
                [ 0.1594,  0.3825]],
                [[ 0.0268,  1.0783],
                [ 0.5041, -1.3025],
                [ 0.6568,  0.7048]],
                [[-0.2688,  1.0543],
                [-0.0315, -0.9016],
                [ 0.3943,  0.5347]],
                [[-0.6066,  1.0268],
                [-0.5945, -0.4801],
                [ 0.1594,  0.3825]]], grad_fn=<BinaryReduceBackward>)
        """
        super().__init__()
        self.num_heads = num_heads
        self.out_feats = out_feats
        self.in_feats = in_feats
        self.allow_zero_in_degree = allow_zero_in_degree

        self.is_homo_conv = True

        # Handling separate transformations for bipartite graphs
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(in_feats[0], out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(in_feats[1], out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(in_feats, out_feats * num_heads, bias=False)

        self.attn_l = nn.Parameter(torch.Tensor(1, num_heads, out_feats))
        self.attn_r = nn.Parameter(torch.Tensor(1, num_heads, out_feats))
        self.feat_drop = nn.Dropout(dropout)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.softmax = nn.Softmax(dim=1)

        self.residual = residual
        self.activation = activation

        if residual:
            if isinstance(in_feats, tuple):
                if in_feats[1] != out_feats * num_heads:
                    self.res_fc = nn.Linear(in_feats[1], out_feats * num_heads,
                                            bias=bias)
                else:
                    self.res_fc = nn.Identity()
            else:
                if in_feats != out_feats * num_heads:
                    self.res_fc = nn.Linear(in_feats, out_feats * num_heads, bias=bias)
                else:
                    self.res_fc = nn.Identity()
        else:
            self.register_buffer('res_fc', None)

        if bias and not (isinstance(self.res_fc, nn.Linear) and residual):
            self.bias = nn.Parameter(torch.Tensor(num_heads * out_feats))
        else:
            self.register_buffer('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        r"""
        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
            if self.res_fc.bias is not None:
                nn.init.constant_(self.res_fc.bias, 0)

    def forward(self, g, h, edge_weight=None, get_attention=False):
        r"""
        Description
        -----------
        Compute graph attention network layer.

        Parameters
        ----------
        h : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, *, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, *, D_{in_{src}})` and :math:`(N_{out}, *, D_{in_{dst}})`.
        g : torch.sparse_coo_tensor
            The graph.
        edge_weight : torch.Tensor, optional
            A 1D tensor of edge weight values.  Shape: :math:`(|E|,)`.
        get_attention : bool, optional
            Whether to return the attention values. Default to False.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, *, H, D_{out})` where :math:`H`
            is the number of heads, and :math:`D_{out}` is size of output feature.
        torch.Tensor, optional
            The attention values of shape :math:`(E, *, H, 1)`, where :math:`E` is the number of
            edges. This is returned only when :attr:`get_attention` is ``True``.

        Raises
        ------
        ValueError
            If there are 0-in-degree nodes in the input graph, it will raise ValueError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.
        """
        # if not self.allow_zero_in_degree and (g.sum(dim=1) == 0).any():
        if not self.allow_zero_in_degree and (torch.sparse.sum(g) == 0).any():
            raise ValueError(
                "There are 0-in-degree nodes in the graph, output for those nodes will be invalid. Adding self-loops will resolve the issue.")

        if isinstance(h, tuple):
            h_src = self.feat_drop(h[0])
            h_dst = self.feat_drop(h[1])
            if not hasattr(self, 'fc_src'):
                Wh_src = self.fc(h_src).view(-1, self.num_heads, self.out_feats)
                Wh_dst = self.fc(h_dst).view(-1, self.num_heads, self.out_feats)
            else:
                Wh_src = self.fc_src(h_src).view(-1, self.num_heads, self.out_feats)
                Wh_dst = self.fc_dst(h_dst).view(-1, self.num_heads, self.out_feats)
        else:
            h = self.feat_drop(h)
            Wh = self.fc(h).view(-1, self.num_heads, self.out_feats)
            Wh_src = Wh_dst = Wh

        # Compute attention scores
        e_l = (Wh_src * self.attn_l).sum(dim=-1)  # Shape (N, 1, num_heads)
        e_r = (Wh_dst * self.attn_r).sum(dim=-1)  # Shape (1, N, num_heads)

        e = e_l[g._indices()[0, :].t(), :] + e_r[g._indices()[1, :].t(), :]
        e = self.leaky_relu(e)

        # Compute edge softmax, we use torch.sparse.softmax here
        # construct a sparse tensor for edge attention scores
        # shape （N, N, num_heads)
        shape = g.shape  # shape (N, N)
        # add num_heads dimension to the shape
        shape = shape + (self.num_heads,)
        attention = torch.sparse_coo_tensor(g._indices(), e, shape)
        attention = torch.sparse.softmax(attention, dim=0)

        attention = attention.coalesce().values()

        attention = self.attn_drop(attention)

        if edge_weight is not None:
            edge_weight = edge_weight.unsqueeze(1).expand(-1, self.num_heads)
            # edge_weight = torch.sparse_coo_tensor(g._indices(), edge_weight, shape)
            attention = attention * edge_weight

        h_prime = []
        for i in range(self.num_heads):
            head_values = attention[:, i]
            g = g.coalesce()
            attention_i = torch.sparse_coo_tensor(g.indices(), head_values,
                                                  g.shape)
            # h_prime.append(torch.spmm(attention_i, Wh_dst[:, i]))
            h_prime.append(torch.spmm(attention_i.t(), Wh_src[:, i]))
        h_prime = torch.stack(h_prime, dim=1)

        if self.residual:
            if isinstance(h, tuple):
                resval = self.res_fc(h[1]).view(-1, self.num_heads, self.out_feats)
            else:
                resval = self.res_fc(h).view(-1, self.num_heads, self.out_feats)
            h_prime += resval

        if self.bias is not None:
            h_prime += self.bias.view(1, -1, self.out_feats)

        if self.activation:
            h_prime = self.activation(h_prime)

        if get_attention:
            return h_prime, attention
        else:
            return h_prime


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
        self.softmax = nn.Softmax(dim=0)

    def forward(self, z, attention=False):
        w = self.project(z)  # (M, 1)
        if attention:
            beta_copy = w
        w = w.mean(0)
        beta = self.softmax(w)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)

        if attention:
            return (beta * z).sum(1), beta_copy

        return (beta * z).sum(1)  # (N, D * K)


class HANLayer(nn.Module):
    """
    HAN layer.

    Arguments
    ---------
    num_meta_paths : number of homogeneous graphs generated from the metapaths.
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability

    Inputs
    ------
    gs : list[torch.SparseTensor]
        List of graphs
    h : tensor
        Input features

    Outputs
    -------
    tensor
        The output feature
    """

    def __init__(self, num_meta_paths, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()

        self.is_hetero_conv = True

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(num_meta_paths):
            self.gat_layers.append(GATConv(in_size, out_size, layer_num_heads,
                                           dropout, dropout, activation=F.elu,
                                           allow_zero_in_degree=True))
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.num_meta_paths = num_meta_paths

    def forward(self, gs, h, gat_attention=False, semantic_attention=False):
        semantic_embeddings = []
        if not gat_attention:
            for i, g in enumerate(gs):
                if isinstance(h, tuple) or isinstance(h, list):
                    semantic_embeddings.append(
                        self.gat_layers[i](g, h[i]).flatten(1))
                else:
                    semantic_embeddings.append(
                        self.gat_layers[i](g, h).flatten(1))  # (N, D * K)
            semantic_embeddings = torch.stack(semantic_embeddings,
                                              dim=1)  # (N, M, D * K)
        else:
            gat_attention_weights = []
            for i, g in enumerate(gs):
                if isinstance(h, tuple) or isinstance(h, list):
                    semantic_embedding, attention = self.gat_layers[i](g, h[i],
                                                                       get_attention=True)
                else:
                    semantic_embedding, attention = self.gat_layers[i](g, h,
                                                                       get_attention=True)
                semantic_embeddings.append(semantic_embedding.flatten(1))
                gat_attention_weights.append(attention)
            semantic_embeddings = torch.stack(semantic_embeddings, dim=1)

        if semantic_attention:
            output, beta = self.semantic_attention(semantic_embeddings, attention=True)
            if not gat_attention:
                return output, beta
            return output, beta, gat_attention_weights

        if not gat_attention:
            return self.semantic_attention(semantic_embeddings)  # (N, D * K)
        return self.semantic_attention(semantic_embeddings), gat_attention_weights


class HAN(BaseModel):
    """
    HAN model.
    from `Heterogeneous Graph Attention Network
    <https://arxiv.org/pdf/1903.07293.pdf>`__
    """

    # > added for pge
    def embedding(self, custom_forward_fn):
        """
        Get the output of the last hidden layer (before the final classification layer)
        """
        g, h = custom_forward_fn(self) if callable(custom_forward_fn) else custom_forward_fn
        
        # > go through all the HAN layers but the final prediction layer
        for gnn in self.layers:
            h = gnn(g, h)
        
        # > return the output of the last HAN layer (embeddings)
        # > shape: (N, hidden_units * num_heads[-1])
        return h

    def __init__(self, config, dataset):
        super(HAN, self).__init__(config, dataset)
        assert self.dataset.hetero is True, "HAN only supports heterogeneous graph."
        assert isinstance(self.dataset,
                          NodeClassificationDataset), "HAN only supports node classification task."
        self.model_name = 'HAN'
        self.dataset_adaptation()
        self.prepare_modules()
        self.to(self.device)
        self.support_multi_features = True

    def prepare_modules(self):
        self.layers = nn.ModuleList()
        self.layers.append(
            HANLayer(num_meta_paths=self.num_meta_paths,
                     in_size=self.dataset.num_features,
                     out_size=self.config['hidden_units'],
                     layer_num_heads=self.config['num_heads'][0],
                     dropout=self.config['dropout']))
        for l in range(1, len(self.config['num_heads'])):
            self.layers.append(HANLayer(num_meta_paths=self.num_meta_paths,
                                        in_size=self.config['hidden_units'] *
                                                self.config['num_heads'][l - 1],
                                        out_size=self.config['hidden_units'],
                                        layer_num_heads=self.config['num_heads'][l],
                                        dropout=self.config['dropout']))
        self.predict = nn.Linear(
            self.config['hidden_units'] * self.config['num_heads'][-1],
            self.dataset.num_classes)

    def dataset_adaptation(self):
        if isinstance(self.dataset, ACM):
            self._dataset_adaptation_acm()
        elif isinstance(self.dataset, DBLP):
            self._dataset_adaptation_dblp()
        elif isinstance(self.dataset, IMDB):
            self._dataset_adaptation_imdb()
        else:
            self._dataset_adaptation_standard()

    def _dataset_adaptation_acm(self):
        self._dataset_adaptation_standard()

    def _dataset_adaptation_dblp(self):
        self._dataset_adaptation_standard()

    def _dataset_adaptation_imdb(self):
        self._dataset_adaptation_standard()

    def _dataset_adaptation_standard(self):
        # read the metapaths used
        self.meta_paths = self.config['meta_paths']
        gs = [self._edges_to_metapath_adjacency(meta_path) for meta_path in
              self.meta_paths]
        # covert to torch tensor (originally scipy csr matrix)
        # use torch.sparse_coo_tensor to avoid converting to dense tensor
        tensor_gs = []
        for i in range(len(gs)):
            g = gs[i]
            g = g.tocoo()
            indices = np.vstack((g.row, g.col))
            values = g.data
            tensor_gs.append(
                torch.sparse_coo_tensor(indices, values, g.shape).to(self.device)
                .coalesce())
        self.gs = tensor_gs
        self.features = torch.tensor(self.dataset.node_features).to(self.device).float()
        self.num_meta_paths = len(self.meta_paths)

        self.train_mask = torch.from_numpy(np.array(self.dataset.labels[0])[:, 0]).type(
            torch.long).to(self.device)
        train_target = torch.from_numpy(np.array(self.dataset.labels[0])[:, 1]).type(
            torch.long).to(self.device)
        self.valid_mask = torch.from_numpy(np.array(self.dataset.labels[1])[:, 0]).type(
            torch.long).to(self.device)
        valid_target = torch.from_numpy(np.array(self.dataset.labels[1])[:, 1]).type(
            torch.long).to(self.device)
        self.test_mask = torch.from_numpy(np.array(self.dataset.labels[2])[:, 0]).type(
            torch.long).to(self.device)
        test_target = torch.from_numpy(np.array(self.dataset.labels[2])[:, 1]).type(
            torch.long).to(self.device)
        target_tensor = torch.zeros(len(self.dataset.node_features), dtype=torch.long,
                                    device=self.device)
        target_tensor[self.train_mask] = train_target
        target_tensor[self.valid_mask] = valid_target
        target_tensor[self.test_mask] = test_target
        self.labels = target_tensor

    def _edges_to_metapath_adjacency(self, meta_path):
        edges = self.dataset.edges
        edge_directions = self.dataset.edge_directions

        if len(meta_path) < 1:
            raise ValueError("Meta path should have at least one node type.")

        if len(meta_path) == 2:
            #index = edge_directions['edge_types'].index([meta_path[0], meta_path[1]])
            # > Yiran edited
            etype = [meta_path[0], meta_path[1]]
            if etype not in edge_directions['edge_types']:
                raise ValueError(f"Edge type {etype} not found in edge_directions.\n"
                                 f"Available types: {edge_directions['edge_types']}")
            index = edge_directions['edge_types'].index(etype)

            return edges[index]

        # we need to use the multiplication of adjacency matrices to achieve the metapath
        for i in range(1, len(meta_path)):
            #index = edge_directions['edge_types'].index(
            #    [meta_path[i - 1], meta_path[i]])
            # > Yiran edited
            etype = [meta_path[i - 1], meta_path[i]]
            if etype not in edge_directions['edge_types']:
                raise ValueError(f"Edge type {etype} not found in edge_directions.\n"
                                 f"Available types: {edge_directions['edge_types']}")
            index = edge_directions['edge_types'].index(etype)

            # > Yiran added
            if index >= len(edges):
                raise IndexError(f"Index {index} is out of range for edges (length={len(edges)})."
                                 f"\nmeta_path = {meta_path}"
                                 f"\ncurrent edge_type = {[meta_path[i - 1], meta_path[i]]}"
                                 f"\nall edge_types = {edge_directions['edge_types']}")


            if i == 1:
                adj = edges[index]
            else:
                adj = adj @ edges[index]

        # set the data is 1
        adj.data = np.ones_like(adj.data)

        return adj

    def standard_input(self):
        return self.gs, self.features

    def forward(self):
        g, h = self.standard_input()

        for gnn in self.layers:
            h = gnn(g, h)

        return self.predict(h)

    def loss(self, outputs):
        return F.cross_entropy(outputs[self.train_mask], self.labels[self.train_mask])

    def custom_loss(self, outputs, mask):
        return F.cross_entropy(outputs[mask], self.labels[mask])

    @staticmethod
    def loss_fn(outputs, pred):
        return F.cross_entropy(outputs, pred)

    def custom_forward(self, handle_fn):
        """
        Custom forward function to allow for custom handling of the input of the model.
        Some explainer and explainer metrics require modification of the input before
        passing it to the model. This function allows for that modification.
        :param handle_fn:
        :return:
        """

        g, h = handle_fn(self)

        for gnn in self.layers:
            h = gnn(g, h)

        return self.predict(h)

    def fit(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'],
                                     weight_decay=self.config['weight_decay'])

        early_stopping = 0
        loss_compared = 1e10

        for epoch in range(self.config['num_epochs']):
            self.train()
            logits = self.forward()
            loss = self.loss(logits)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss = loss.item()

            # evaluate by validation set
            self.eval()
            logits = self.forward()
            loss = self.custom_loss(logits, self.valid_mask)

            metrics, results = self.evaluate(logits, self.labels, self.valid_mask)

            display_str = f"Epoch {epoch} | Train Loss: {train_loss} | Validation Loss: {loss.item()} | "
            for metric in metrics:
                display_str += f"{metric}: {results[metric]} | "
            print(display_str)

            if loss.item() < loss_compared:
                self.temp_state_dict = copy.deepcopy(self.state_dict())
                early_stopping = 0
                loss_compared = loss.item()
            else:
                early_stopping += 1

            if early_stopping >= self.config['patience']:
                break

        self.load_state_dict(self.temp_state_dict)

    def evaluate(self, logits, labels, mask):
        logits_computed = logits[mask]
        labels_computed = labels[mask]
        result = {}
        for metric in self.metrics:
            result[metric] = node_classification_support_metrics[metric](
                logits_computed,
                labels_computed)
        return self.metrics, result

    def get_summary(self):
        self.summary = {}
        logits = self.forward()
        metrics, results = self.evaluate(logits, self.labels, self.test_mask)
        for metric in metrics:
            self.summary[metric] = results[metric]

        if self.config['summary_path']:
            self.save_summary()

        return self.summary

    def save_summary(self):
        if self.config['summary_path'] is None:
            return
        import os
        os.makedirs(os.path.dirname(self.config['summary_path']), exist_ok=True)
        with open(self.config['summary_path'], 'w') as f:
            json.dump(self.summary, f)

    def save(self, path=None, save_dataset_relative=False):
        save_dict = {
            'config': self.config,
            'state_dict': self.state_dict(),
            'metrics': self.metrics,
            # 'device': self.device, # no need to save device, we ensure when loading, the device is cpu
            'meta_paths': self.meta_paths,
            'num_meta_paths': self.num_meta_paths,
            'model_name': self.model_name
        }
        if getattr(self, 'summary', None) is not None:
            save_dict['summary'] = self.summary
        if save_dataset_relative:
            dataset_dict = self.dataset.to_dict()
            filter_keys = ['node_features', 'node_types', 'edges', 'labels',
                           'edge_directions']
            dataset_dict = {k: dataset_dict[k] for k in dataset_dict if
                            k not in filter_keys}
            save_dict['dataset'] = dataset_dict
            other_keys = ['gs', 'features', 'labels', 'train_mask', 'valid_mask',
                          'test_mask']
            for key in other_keys:
                save_dict[key] = getattr(self, key)

        if path is None:
            path = self.config['weight_path']

        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(save_dict, path)

    @classmethod
    def load(cls, path, dataset=None):
        if dataset is None:
            read_dataset_relative = True
        else:
            read_dataset_relative = False

        load_dict = torch.load(path)
        config = load_dict['config']
        if not read_dataset_relative:
            model = cls(config, dataset)
            model.load_state_dict(load_dict['state_dict'])
            model.summary = load_dict.get('summary', None)
        else:
            dataset_dict = load_dict['dataset']
            # not all dataset attributes are saved in the dataset_dict
            # see filter_keys in the save method
            # these have been converted to like gs, features, labels,
            # train_mask, valid_mask, test_mask
            # This way we can save memory
            dataset = ACM.from_dict(dataset_dict)
            model = cls.__new__(cls)
            model.dataset = dataset
            model.config = config
            load_keys = ['metrics', 'meta_paths', 'num_meta_paths',
                         'model_name', 'gs', 'features', 'labels',
                         'train_mask', 'valid_mask', 'test_mask']
            for key in load_keys:
                setattr(model, key, load_dict[key])

            # set the device to cpu
            model.device = 'cpu'
            model.prepare_modules()
            model.load_state_dict(load_dict['state_dict'])
            model.summary = load_dict.get('summary', None)
        return model

    def save_attention(self, path=None, gat_attention=False, semantic_attention=True):
        if path is None:
            path = self.config['attention_path']
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        attentions = []
        g, h = self.standard_input()
        for gnn in self.layers:
            h, *attention = gnn(g, h, gat_attention=gat_attention,
                                semantic_attention=semantic_attention)
            attentions.append(attention)
        torch.save(attentions, path)

    def custom_forward_meta_path_mask(self, input_handle_fn, meta_path_weight_handle_fn
                                      ):
        # we need to use hook to process the meta path weight
        # for han, meta path weight is the attention weights (output of the softmax layer
        # of the semantic attention)
        # meta_path_weight_handle_fn: give the meta path weight, return the modified weight

        def forward_hook(module, input, output):
            meta_path_weight = meta_path_weight_handle_fn(output)
            return meta_path_weight

        hooks = []
        for gnn in self.layers:
            hooks.append(
                gnn.semantic_attention.softmax.register_forward_hook(forward_hook))

        output = self.custom_forward(input_handle_fn)

        for hook in hooks:
            hook.remove()

        return output

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.device = self.predict.weight.device
        # move data to device
        self.gs = [g.to(self.device) for g in self.gs]
        self.features = self.features.to(self.device)
        self.train_mask = self.train_mask.to(self.device)
        self.valid_mask = self.valid_mask.to(self.device)
        self.test_mask = self.test_mask.to(self.device)
        self.labels = self.labels.to(self.device)
