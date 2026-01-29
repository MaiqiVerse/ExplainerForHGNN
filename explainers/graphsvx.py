import os
import math
from typing import Tuple
from copy import deepcopy
from itertools import combinations
import random

import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score 
from torch.autograd import Variable
import numpy as np
import scipy.special
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from torch import Tensor
from scipy.special import comb

from .explainer import Explainer, ExplainerCore
from .explanation import NodeExplanation, NodeExplanationCombination
from .prepare_explanation_for_node_scores import standard_explanation
from .prepare_explanation_for_node_dataset_scores import \
    prepare_explanation_fn_for_node_dataset_scores
from .prepare_combined_explanation_for_node_dataset_scores import \
    prepare_combined_explanation_fn_for_node_dataset_scores
from .node_dataset_scores import node_dataset_scores


class LinearRegressionModel(nn.Module):
    """A simple linear regression
    """    
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        y_pred =self.linear1(x)
        return y_pred

class GraphSVXCore(ExplainerCore):

    def __init__(self, config):
        super().__init__(config)

        self.num_samples = config.get("num_samples", 10)
        self.n_hop = config.get("n_hop", 2)

        self.info = config.get("info", False)
        self.multiclass = config.get("multiclass", False)
        self.fullempty = config.get("fullempty", None)
        self.S = config.get("S", 3)

        self.args_hv = config.get("args_hv", "compute_pred_hetero")
        self.args_feat = config.get("args_feat", "Expectation")
        self.args_coal = config.get("args_coal", "Smarter")
        self.args_g = config.get("args_g", "WLR_sklearn")

        self.regu = config.get("regu", 0)
        self.vizu = config.get("vizu", False)


        self.neighbours = None  #  nodes considered
        self.F = 0  # number of features considered
        self.M = None  # number of features and nodes considered
        self.base_values = []

    @torch.no_grad()
    def explain(self, model, **kwargs):

        self.model = model
        self.model.eval()

        self.node_id = kwargs.get("node_id")
        if self.node_id is None:
            raise ValueError("node_id is required")
        
        print(f"[GraphSVXCore] >>> Starting explanation for node_id = {self.node_id}")
        
        # 1) Extract local subgraphs (per meta-path)
        gs, features = self.extract_neighbors_input()
        self.gs = [g.to(self.device_string) for g in gs]
        self.features = features.to(self.device_string)
        self.target_local = self.recovery_dict[self.node_id]

        # 2) Compute f_full ~ prob
        self.prob, self.target_class = self._compute_ffull(self.gs, self.features)
        # 3) Feature selection
        feat_idx, discarded_feat_idx = self.feature_selection(self.target_local)
        # 4) number of players computation per meta-path
        args_K = self.S
        z_list = []

        for mp_idx in range(len(self.gs)):
            g = self.gs[mp_idx].coalesce()
            idx = g.indices()
            src, dst = idx[0], idx[1]

            t = self.target_local
            mask = (src == t) | (dst == t)

            nbrs_from_src = dst[src[mask] == t]
            nbrs_from_dst = src[dst[mask] == t]

            neighbours_mp = torch.cat([nbrs_from_src, nbrs_from_dst], dim=0)
            neighbours_mp = torch.unique(neighbours_mp)
            neighbours_mp = neighbours_mp[neighbours_mp != t]

            D_m = neighbours_mp.numel()

            if D_m <= 1:
                z_list.append({
                    "mp_idx": mp_idx,
                    "z": None,
                    "weights": None,
                    "neighbours": neighbours_mp,
                    "D": D_m,
                    "phi": torch.zeros(D_m, device=self.device_string)
                })
                continue

            F_local = self.F if self.regu != 0 else 0
            D_local = D_m if self.regu != 1 else 0
            M_local = F_local + D_local

            if M_local == 0:
                continue

            # IMPORTANT: set GraphSVX internal state JUST FOR THIS CALL
            self.F = F_local
            self.D = D_local
            self.M = M_local


            args_K = self.S

            z_, weights = self.mask_generation(
                self.num_samples,
                self.args_coal,
                args_K,
                D_local,
                self.info,
                self.regu
            )


            z_list.append({
                "mp_idx": mp_idx,
                "z": z_,
                "weights": weights,
                "neighbours": neighbours_mp,
                "D": D_m
            })

            # Discard full and empty coalition if specified
            if self.fullempty:
                weights[weights == 1000] = 0
        # 5) # --- GRAPH GENERATOR ---
        fz_list = []

        for item in z_list:
            mp_idx = item["mp_idx"]
            z_ = item["z"]
            neighbours_mp = item["neighbours"]
            D_m = item["D"]

            # Skip metapaths that were skipped earlier
            if z_ is None or D_m == 0:
                fz_list.append({
                    "mp_idx": mp_idx,
                    "fz": None
                })
                continue

            fz = eval('self.' + self.args_hv)(
                mp_idx=mp_idx,
                node_index=self.target_local,
                num_samples=self.num_samples,
                D=D_m,
                z_=z_,
                feat_idx=feat_idx,
                neighbours_mp=neighbours_mp,
                args_K=args_K,
                args_feat=self.args_feat,
                discarded_feat_idx=discarded_feat_idx,
                multiclass=self.multiclass,
                true_pred=self.target_class,
            )

            fz_list.append({
                "mp_idx": mp_idx,
                "fz": fz
            })

        # 6) EXPLANATION GENERATOR (per metapath)
        phi_list = []

        for item, fz_item in zip(z_list, fz_list):
            mp_idx = item["mp_idx"]

            z_ = item["z"]
            weights = item["weights"]
            fz = fz_item["fz"]
            D_m = item["D"]

            # Skip metapaths that were skipped earlier
            if z_ is None or fz is None or D_m == 0:
                phi_list.append({
                    "mp_idx": mp_idx,
                    "phi": torch.zeros(D_m, device=self.device_string),
                    "base_value": None,
                    "neighbours": item["neighbours"]
                })
                continue

            phi, base_value = eval('self.' + self.args_g)(
                z_,
                weights,
                fz,
                self.multiclass,
                self.info
            )

            phi_list.append({
                "mp_idx": mp_idx,
                "phi": phi,
                "base_value": base_value,
                "neighbours": item["neighbours"]
            })

        self.phi_list = phi_list

        # GraphSVX structure-only explanation
        self.edge_mask = self.edge_mask_for_output
        self.feature_mask = None

        return self.construct_explanation()

    @property
    def edge_mask_for_output(self):

        edge_masks = []

        for mp_idx, item in enumerate(self.phi_list):
            phi = item["phi"]
            neighbours = item["neighbours"]

            g = self.gs[mp_idx].coalesce()
            idx = g.indices()
            src, dst = idx[0], idx[1]

            # Initialize edge mask
            em = torch.zeros(
                src.size(0),
                device=self.device_string,
                dtype=torch.float
            )

            if phi is None or neighbours.numel() == 0:
                edge_masks.append(em)
                continue

            phi = torch.as_tensor(phi, device=self.device_string).flatten()

            # Assign φ(u) to all edges incident to u
            for j, u in enumerate(neighbours):
                u = int(u.item())
                mask = (src == u) | (dst == u)
                em[mask] = phi[j]

            edge_masks.append(em)

        return edge_masks

    @property
    def feature_mask_for_output(self):
        if self.regu == 0:
            return None

    # ==================================================================
    # Build final explanation (for fidelity)
    # ==================================================================
    def construct_explanation(self):
        explanation = NodeExplanation()
        explanation = standard_explanation(explanation, self)

        for metric in self.config['eval_metrics']:
            prepare_explanation_fn_for_node_dataset_scores[metric](explanation, self)

        self.explanation = explanation
        return explanation

    # =========================================================
    #       Node id mapping (global → local in subgraph)
    # =========================================================
    def mapping_node_id(self):
        if getattr(self, "mapped_node_id", None) is not None:
            return self.mapped_node_id

        if not self.config.get("extract_neighbors", True):
            self.mapped_node_id = self.node_id
        else:
            self.mapped_node_id = self.recovery_dict[self.node_id]

        return self.mapped_node_id

    def get_custom_input_handle_fn(self, masked_gs=None, feature_mask=None):
        def handle_fn(model):
            if model is None:
                model = self.model
            gs, features = self.extract_neighbors_input()

            if masked_gs is not None:
                gs = []
                for g in masked_gs:
                    mask = g.values() != 0
                    indices = g.indices()[:, mask]
                    value = g.values()[mask]
                    shape = g.size()
                    gs.append(torch.sparse_coo_tensor(
                        indices, value, shape))

                gs = [g.to(self.device_string) for g in gs]
            if feature_mask is not None:
                feature_mask_device = feature_mask.to(self.device_string)
                features = features * feature_mask_device
            return gs, features

        return handle_fn


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
    
    def _compute_ffull(self, gs, features):
        logits = self.model.custom_forward(lambda _: (gs, features))
        node_logits = logits[self.target_local]
        cls = node_logits.argmax().item()
        prob = F.softmax(node_logits, dim=0)[cls].item()
        return prob, cls

    def feature_selection(self, target_local):
        """ Select features who truly impact prediction
        Others will receive a 0 shapley value anyway 
        """
        #Only consider relevant features in explanations
        discarded_feat_idx = []

        # Select features whose value is different from dataset mean value
        mean = self.model.features.mean(dim=0).to(self.device_string)
        std  = self.model.features.std(dim=0).to(self.device_string)

        node_feat = self.features[self.target_local]

        mean_subgraph = torch.where(
            node_feat >= mean - 0.25 * std,
            node_feat,
            torch.ones_like(node_feat) * 100
        )

        mean_subgraph = torch.where(
            node_feat <= mean + 0.25 * std,
            mean_subgraph,
            torch.ones_like(node_feat) * 100
        )

        feat_idx = (mean_subgraph == 100).nonzero(as_tuple=False)
        discarded_feat_idx = (mean_subgraph != 100).nonzero(as_tuple=False)
        self.F = feat_idx.shape[0]

        return feat_idx, discarded_feat_idx


    def mask_generation(self, num_samples, args_coal, args_K, D, info, regu):
            """ Applies selected mask generator strategy 

            Args:
                num_samples (int): number of samples for GraphSVX 
                args_coal (str): mask generator strategy 
                args_K (int): size param for indirect effect 
                D (int): number of nodes considered after selection
                info (bool): print information or not 
                regu (int): balances importance granted to nodes and features

            Returns:
                [tensor] (num_samples, M): dataset of samples/coalitions z' 
                [tensor] (num_samples): vector of kernel weights corresponding to samples 
            """
            if args_coal == 'SmarterSeparate' or args_coal == 'NewSmarterSeparate':
                weights = torch.zeros(num_samples, dtype=torch.float64)
                if self.F==0 or D==0:
                    num = int(num_samples * self.F/self.M)
                elif regu != None:
                    num = int(num_samples * regu)
                    #num = int( num_samples * ( self.F/self.M + ((regu - 0.5)/0.5)  * (self.F/self.M) ) )    
                else: 
                    num = int(0.5* num_samples/2 + 0.5 * num_samples * self.F/self.M)
                # Features only
                z_bis = eval('self.' + args_coal)(num, args_K, 1)  
                z_bis = z_bis[torch.randperm(z_bis.size()[0])]
                s = (z_bis != 0).sum(dim=1)
                weights[:num] = self.shapley_kernel(s, self.F)
                z_ = torch.zeros(num_samples, self.M)
                z_[:num, :self.F] = z_bis
                # Node only
                z_bis = eval('self.' + args_coal)(
                    num_samples-num, args_K, 0)  
                z_bis = z_bis[torch.randperm(z_bis.size()[0])]
                s = (z_bis != 0).sum(dim=1)
                weights[num:] = self.shapley_kernel(s, D)
                z_[num:, :] = torch.ones(num_samples-num, self.M)
                z_[num:, self.F:] = z_bis

            else:
                # If we choose to sample all possible coalitions
                if args_coal == 'All':
                    num_samples = min(10000, 2**self.M)

                # Coalitions: sample num_samples binary vectors of dimension M
                z_ = eval('self.' + args_coal)(num_samples, args_K, regu)

                # Shuffle them 
                z_ = z_[torch.randperm(z_.size()[0])]

                # Compute |z| for each sample z: number of non-zero entries
                s = (z_ != 0).sum(dim=1)

                # GraphSVX Kernel: define weights associated with each sample 
                weights = self.shapley_kernel(s, self.M)
                
            return z_, weights



    def SmarterSeparate(self, num_samples, args_K, regu):
        """Default mask sampler
        Generates feature mask and node mask independently
        Favours masks with a high weight + efficient space allocation algorithm

        Args:
            num_samples (int): number of masks desired 
            args_K (int): maximum size of masks favoured
            regu (binary): nodes or features 

        Returns:
            tensor: dataset of samples
        """
        if regu == None:
            z_ = self.Smarter(num_samples, args_K, regu)
            return z_

        # Favour features - special coalitions don't study node's effect
        elif regu > 0.5:
            # Define empty and full coalitions
            M = self.F
            z_ = torch.ones(num_samples, M)
            z_[1::2] = torch.zeros(num_samples//2, M)
            # z_[1, :] = torch.empty(1, self.M).random_(2)
            i = 2
            k = 1
            # Loop until all samples are created
            while i < num_samples:
                # Look at each feat/nei individually if have enough sample
                # Coalitions of the form (All nodes/feat, All-1 feat/nodes) & (No nodes/feat, 1 feat/nodes)
                if i + 2 * M < num_samples and k == 1:
                    z_[i:i+M, :] = torch.ones(M, M)
                    z_[i:i+M, :].fill_diagonal_(0)
                    z_[i+M:i+2*M, :] = torch.zeros(M, M)
                    z_[i+M:i+2*M, :].fill_diagonal_(1)
                    i += 2 * M
                    k += 1

                else:
                    # Split in two number of remaining samples
                    # Half for specific coalitions with low k and rest random samples
                    samp = i + 9*(num_samples - i)//10
                    #samp = num_samples
                    while i < samp and k <= min(args_K, M):
                        # Sample coalitions of k1 neighbours or k1 features without repet and order.
                        L = list(combinations(range(M), k))
                        random.shuffle(L)
                        L = L[:samp+1]

                        for j in range(len(L)):
                            # Coalitions (All nei, All-k feat) or (All feat, All-k nei)
                            z_[i, L[j]] = torch.zeros(k)
                            i += 1
                            # If limit reached, sample random coalitions
                            if i == samp:
                                return z_
                            # Coalitions (No nei, k feat) or (No feat, k nei)
                            z_[i, L[j]] = torch.ones(k)
                            i += 1
                            # If limit reached, sample random coalitions
                            if i == samp:
                                return z_
                        k += 1

                    # Sample random coalitions
                    z_[i:, :] = torch.empty(num_samples-i, M).random_(2)
                    return z_
            return z_

        # Favour neighbour
        else:
            # Define empty and full coalitions
            M = self.M - self.F
            # self.F = 0
            z_ = torch.ones(num_samples, M)
            z_[1::2] = torch.zeros(num_samples//2, M)
            i = 2
            k = 1
            # Loop until all samples are created
            while i < num_samples:
                # Look at each feat/nei individually if have enough sample
                # Coalitions of the form (All nodes/feat, All-1 feat/nodes) & (No nodes/feat, 1 feat/nodes)
                if i + 2 * M < num_samples and k == 1:
                    z_[i:i+M, :] = torch.ones(M, M)
                    z_[i:i+M, :].fill_diagonal_(0)
                    z_[i+M:i+2*M, :] = torch.zeros(M, M)
                    z_[i+M:i+2*M, :].fill_diagonal_(1)
                    i += 2 * M
                    k += 1

                else:
                    # Split in two number of remaining samples
                    # Half for specific coalitions with low k and rest random samples
                    #samp = i + 9*(num_samples - i)//10
                    samp = num_samples
                    while i < samp and k <= min(args_K, M):
                        # Sample coalitions of k1 neighbours or k1 features without repet and order.
                        L = list(combinations(range(0, M), k))
                        random.shuffle(L)
                        L = L[:samp+1]

                        for j in range(len(L)):
                            # Coalitions (All nei, All-k feat) or (All feat, All-k nei)
                            z_[i, L[j]] = torch.zeros(k)
                            i += 1
                            # If limit reached, sample random coalitions
                            if i == samp:
                                z_[i:, :] = torch.empty(num_samples-i, M).random_(2)
                                return z_
                            # Coalitions (No nei, k feat) or (No feat, k nei)
                            z_[i, L[j]] = torch.ones(k)
                            i += 1
                            # If limit reached, sample random coalitions
                            if i == samp:
                                z_[i:, :] = torch.empty(num_samples-i, M).random_(2)
                                return z_
                        k += 1

                    # Sample random coalitions
                    z_[i:, :] = torch.empty(num_samples-i, M).random_(2)
                    #print(f"randomly sampled: {num_samples-i}")
                    return z_
            return z_

    def shapley_kernel(self, s, M):
        """ Computes a weight for each newly created sample 

        Args:
            s (tensor): contains dimension of z for all instances
                (number of features + neighbours included)
            M (tensor): total number of features/nodes in dataset

        Returns:
                [tensor]: shapley kernel value for each sample
        """
        shapley_kernel = []

        for i in range(s.shape[0]):
            a = s[i].item()
            if a == 0 or a == M:
                # Enforce high weight on full/empty coalitions
                shapley_kernel.append(1000)
            elif scipy.special.binom(M, a) == float('+inf'):
                # Treat specific case - impossible computation
                shapley_kernel.append(1/ (M**2))
            else:
                shapley_kernel.append(
                    (M-1)/(scipy.special.binom(M, a)*a*(M-a)))

        shapley_kernel = np.array(shapley_kernel)
        shapley_kernel = np.where(shapley_kernel<1.0e-40, 1.0e-40,shapley_kernel)
        return torch.tensor(shapley_kernel)

    def compute_pred_hetero(
        self,
        mp_idx: int,
        node_index: int,
        num_samples: int,
        D: int,
        z_: torch.Tensor,
        feat_idx: torch.Tensor,
        neighbours_mp: torch.Tensor,
        args_K: int,
        args_feat: str,
        discarded_feat_idx: torch.Tensor,
        multiclass: bool,
        true_pred: int,
    ):

        device = self.device_string

        neighbours_mp = neighbours_mp[neighbours_mp != node_index]
        F_local = z_.shape[1] - D   # IMPORTANT FIX

        if args_feat == "Null":
            av_feat_values = torch.zeros(self.features.size(1), device=device)
        else:
            av_feat_values = self.model.features.mean(dim=0).to(device)

        if multiclass:
            fz = None   # lazy allocation
        else:
            fz = torch.zeros(num_samples, device=device)

        gs_orig = [g.coalesce() for g in self.gs]
        g_mp = gs_orig[mp_idx]

        idx = g_mp.indices()   # (2, E)
        vals = g_mp.values()
        src, dst = idx[0], idx[1]


        for i in range(num_samples):
            ex_feat = []
            for j in range(F_local):
                if z_[i, j].item() == 0:
                    ex_feat.append(int(feat_idx[j].item()))

            ex_nei = []
            for j in range(D):
                if z_[i, F_local + j].item() == 0:
                    ex_nei.append(int(neighbours_mp[j].item()))

            Xp = self.features.clone()
            if len(ex_feat) > 0:
                ex_feat_t = torch.tensor(ex_feat, device=device, dtype=torch.long)
                Xp[node_index, ex_feat_t] = av_feat_values[ex_feat_t]


            if len(ex_nei) > 0:
                ex_nei_t = torch.tensor(ex_nei, device=device, dtype=torch.long)

                keep = (~torch.isin(src, ex_nei_t)) & (~torch.isin(dst, ex_nei_t))

                masked_idx = idx[:, keep]
                masked_vals = vals[keep]

                masked_g = torch.sparse_coo_tensor(
                    masked_idx,
                    masked_vals,
                    size=g_mp.size(),
                    device=device,
                ).coalesce()
            else:
                masked_g = g_mp

            # ---------- compose gs' ----------
            gs_masked = list(gs_orig)
            gs_masked[mp_idx] = masked_g

            # ---------- forward HAN_GCN ----------
            logits = self.model.custom_forward(lambda _: (gs_masked, Xp))
            node_logits = logits[node_index]

            if multiclass:
                if fz is None:
                    fz = torch.zeros(
                        (num_samples, node_logits.numel()),
                        device=device
                    )
                fz[i] = F.softmax(node_logits, dim=0)
            else:
                fz[i] = F.softmax(node_logits, dim=0)[true_pred]

        return fz




    def WLR_sklearn(self, z_, weights, fz, multiclass, info):
        """Train a weighted linear regression

        Args:
            z_ (torch.tensor): dataset
            weights (torch.tensor): weights of each sample
            fz (torch.tensor): predictions for z_ 

        Return:
            tensor: parameters of explanation model g
        """
        # Convert to numpy
        weights = weights.detach().numpy()
        z_ = z_.detach().numpy()
        fz = fz.detach().cpu().numpy()

        # Fit weighted linear regression
        reg = LinearRegression()
        reg.fit(z_, fz, weights)
        y_pred = reg.predict(z_)

        # Assess perf
        if info:
            print('weighted r2: ', reg.score(z_, fz, sample_weight=weights))
            print('r2: ', r2_score(fz, y_pred))

        # Coefficients
        phi = reg.coef_
        base_value = reg.intercept_

        return phi, base_value



class GraphSVX(Explainer):

    def __init__(self, config):
        super().__init__(config)

    def explain(self, model, **kwargs):
        self.model = model

        if not getattr(model.dataset, "single_graph", True):
            raise ValueError("Only single-graph datasets are supported.")

        return self._explain_dataset(**kwargs)

    def _explain_dataset(self, **kwargs):
        core_class = self.core_class()
        test_labels = self.model.dataset.labels[2]

        max_nodes = kwargs.get("max_nodes", None)
        if max_nodes is not None:
            test_labels = test_labels[:max_nodes]

        explanations = []
        for idx, _ in test_labels:
            core = core_class(self.config)
            core.to(self.device)
            explanations.append(core.explain(self.model, node_id=idx))

        combined = NodeExplanationCombination(node_explanations=explanations)

        if self.config.get("control_data", None) is not None:
            combined.control_data = self.config["control_data"]

        self._evaluate(combined)
        self._save_summary()

        return self.eval_result

    def _evaluate(self, combined):
        eval_result = {}

        if self.config.get("eval_metrics", None):
            for metric in self.config["eval_metrics"]:
                combined = prepare_combined_explanation_fn_for_node_dataset_scores[metric](
                    combined, self
                )
                eval_result[metric] = node_dataset_scores[metric](combined)

        self.eval_result = eval_result
        return eval_result

    def _save_summary(self):
        path = self.config.get("summary_path", None)
        if path is None:
            return
        import os, json
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.eval_result, f)

    def core_class(self):
        return GraphSVXCore
    

