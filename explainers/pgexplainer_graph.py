import torch
import torch.nn as nn
import torch.nn.functional as F
from .explainer import ExplainerCore


class PGExplainerGraphCore(ExplainerCore):
    def __init__(self, config):
        super().__init__(config)

    def init_params_graph_level(self):
        self.embedding_dim = self.model.embedding_dim  # required attr
        self.elayers = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(self.device)

        self.nodesize = self.model.num_nodes  # required attr
        rc = torch.arange(self.nodesize)
        self.row = rc.repeat(self.nodesize)
        self.col = rc.view(self.nodesize, 1).repeat(1, self.nodesize).reshape(-1)

        self.diag_mask = (torch.ones(self.nodesize, self.nodesize) - torch.eye(self.nodesize)).float().to(self.device)
        self.mask_act = "sigmoid"
        self.tmp = self.config.get("tmp", 0.1)

    def concrete_sample(self, log_alpha, beta=1.0, training=True):
        eps = 1e-6
        if training:
            noise = torch.rand_like(log_alpha).clamp(eps, 1. - eps)
            gate_inputs = (torch.log(noise) - torch.log(1.0 - noise) + log_alpha) / beta
            return torch.sigmoid(gate_inputs)
        else:
            return torch.sigmoid(log_alpha)

    def forward_graph_level(self):
        x = self.model.input_x
        embed = self.model.embed
        adj = self.model.input_adj
        label = self.model.input_label

        self.label = torch.argmax(label, dim=-1)

        f1 = embed[self.row]
        f2 = embed[self.col]
        f12 = torch.cat([f1, f2], dim=-1)
        h = self.elayers(f12)
        self.values = h.view(-1)

        values = self.concrete_sample(self.values, beta=self.tmp, training=True)

        indices = torch.stack([self.row, self.col], dim=0)
        mask = torch.sparse.FloatTensor(indices, values, torch.Size([self.nodesize, self.nodesize])).to_dense().to(self.device)
        mask = (mask + mask.t()) / 2
        self.mask = mask

        masked_adj = adj * mask * self.diag_mask
        self.masked_adj = masked_adj

        x = x.unsqueeze(0)
        adj = masked_adj.unsqueeze(0)

        output = self.model((x, adj))
        pred = F.softmax(output, dim=-1)

        self.pred = pred
        return self.get_loss_graph_level(pred)

    def get_loss_graph_level(self, pred):
        logit = pred[0][self.label]
        pred_loss = -torch.log(logit + 1e-6)

        mask = self.mask
        if self.mask_act == "sigmoid":
            mask = torch.sigmoid(mask)
        elif self.mask_act == "ReLU":
            mask = F.relu(mask)

        coff_size = self.config.get("coff_size", 0.005)
        coff_ent  = self.config.get("coff_ent", 0.1)
        #size_loss = args.coff_size * torch.sum(mask)
        size_loss = coff_size * torch.sum(mask)

        mask = mask * 0.99 + 0.005
        ent_loss = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        mask_ent_loss = args.coff_ent * torch.mean(ent_loss)

        return pred_loss + size_loss + mask_ent_loss

