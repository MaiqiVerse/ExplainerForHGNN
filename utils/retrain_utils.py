import copy
import torch
import torch.nn as nn
import sklearn.metrics as metrics

from explainers.explainer import ExplainerCore


class Retrain:
    def __init__(self, explainer, explainer_nodes, explanations):
        self.explainer = explainer
        self.explanations = explanations
        self.explainer_nodes = explainer_nodes
        self.model = copy.deepcopy(self.explainer.model)

    def forward(self, labels):

        model = self.model

        result_collections = []
        label_collections = []

        for idx, label in labels:
            explainer = self.explainer_nodes[idx]
            node_explanation = self.explanations[idx]
            result_collections.append(model.custom_forward(explainer.get_custom_input_handle_fn(
                node_explanation.masked_gs_hard,
                node_explanation.feature_mask_hard))[
                                          explainer.mapping_node_id()])
            label_collections.append(label)

        result_collections = torch.stack(result_collections)
        label_collections = torch.tensor(label_collections, dtype=torch.long, device=result_collections.device)
        loss = self.loss_fn(result_collections, label_collections)
        return loss, result_collections, label_collections

    def fit(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model.config['lr'],
                                     weight_decay=self.model.config['weight_decay'])
        loss_fn = nn.CrossEntropyLoss()
        self.loss_fn = loss_fn

        early_stopping = 0
        loss_compared = 1e10

        train_labels = self.model.dataset.labels[0]
        val_labels = self.model.dataset.labels[1]
        test_labels = self.model.dataset.labels[2]

        for epoch in range(self.model.config['num_epochs']):
            self.model.train()
            loss, _, _ = self.forward(train_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch + 1}/{self.model.config['num_epochs']}, Loss: {loss.item()}")

            self.model.eval()
            with torch.no_grad():
                val_loss, val_result_collections, label_collections = self.forward(val_labels)

                # Macro F1 Score and Micro F1 Score
                val_pred = torch.argmax(val_result_collections, dim=1)
                macro_f1 = metrics.f1_score(label_collections.cpu(), val_pred.cpu(), average='macro')
                micro_f1 = metrics.f1_score(label_collections.cpu(), val_pred.cpu(), average='micro')
                print(f"Validation Loss: {val_loss.item()}, Macro F1: {macro_f1}, Micro F1: {micro_f1}")

                if val_loss.item() < loss_compared:
                    self.temp_state_dict = copy.deepcopy(self.state_dict())
                    loss_compared = val_loss.item()
                    early_stopping = 0
                else:
                    early_stopping += 1

            if early_stopping >= self.model.config['patience']:
                break

        self.model.load_state_dict(self.temp_state_dict)
        self.model.eval()

        with torch.no_grad():
            test_loss, test_result_collections, label_collections = self.forward(test_labels)

            # Macro F1 Score and Micro F1 Score
            test_pred = torch.argmax(test_result_collections, dim=1)
            macro_f1 = metrics.f1_score(label_collections.cpu(), test_pred.cpu(), average='macro')
            micro_f1 = metrics.f1_score(label_collections.cpu(), test_pred.cpu(), average='micro')
            print(f"Retrain Test Loss: {test_loss.item()}, Macro F1: {macro_f1}, Micro F1: {micro_f1}")

        return test_loss, {
            "Macro-F1": macro_f1,
            "Micro-F1": micro_f1,
        }


class CreateAbstract(ExplainerCore):
    def __init__(self, explainer_node):
        super(CreateAbstract, self).__init__(explainer_node.config)
        self.move2cpu = True
        self.catch_necessary_data(explainer_node)
        self.prepare_custom_input_handle_fn()

    def catch_necessary_data(self, explainer_node):
        self.abstract_explainer_node_name = explainer_node.__class__.__name__
        # if self.abstract_explainer_node_name == 'GNNExplainerOriginalCore':
        if self.abstract_explainer_node_name in ['GNNExplainerOriginalCore', 'GNNExplainerMetaCore']:
            self.gs, self.features = explainer_node.extract_neighbors_input()
            self.device = explainer_node.device
            if self.move2cpu:
                self.gs = [g.cpu() for g in self.gs]
                self.features = self.features.cpu()
                self.extract_neighbors_input = lambda: (self.gs.to(self.device), self.features.to(self.device))
            else:
                self.extract_neighbors_input = lambda: (self.gs, self.features)
        elif self.abstract_explainer_node_name == "GradExplainerCore":
            self.gs, self.features = explainer_node.extract_neighbors_input()
            self.device = explainer_node.device
            if self.move2cpu:
                self.gs = [g.cpu() for g in self.gs]
                self.features = self.features.cpu()
                self.extract_neighbors_input = lambda: (self.gs.to(self.device), self.features.to(self.device))
            else:
                self.extract_neighbors_input = lambda: (self.gs, self.features)
            self.support_multi_features = explainer_node.model.support_multi_features
            self.use_meta = explainer_node.config.get('use_meta', False)
        elif self.abstract_explainer_node_name == "SubgraphXCore":
            self.gs, self.features = explainer_node.extract_neighbors_input()
            self.device = explainer_node.device
            if self.move2cpu:
                self.gs = [g.cpu() for g in self.gs]
                self.features = self.features.cpu()
                self.extract_neighbors_input = lambda: (self.gs.to(self.device), self.features.to(self.device))
                self.feature_mask_for_output_out = explainer_node.feature_mask_for_output.cpu()
            else:
                self.extract_neighbors_input = lambda: (self.gs, self.features)
                self.feature_mask_for_output_out = explainer_node.feature_mask_for_output
        else:
            raise NotImplementedError(f"Abstract explainer node {self.abstract_explainer_node_name} is not supported.")

    def prepare_custom_input_handle_fn(self):
        if self.abstract_explainer_node_name in ['GNNExplainerOriginalCore', 'GNNExplainerMetaCore']:
            def used_fn(masked_gs=None, feature_mask=None):
                def handle_fn(model):
                    gs, features = self.extract_neighbors_input()
                    if masked_gs is not None:
                        gs = [g.to(self.device) for g in masked_gs]
                    if feature_mask is not None:
                        features = features * feature_mask.to(self.device)
                    return gs, features

                return handle_fn
        elif self.abstract_explainer_node_name == "GradExplainerCore":
            def used_fn(masked_gs=None, feature_mask=None):
                def handle_fn(model):
                    gs, features = self.extract_neighbors_input()
                    if masked_gs is not None:
                        gs = [g.to(self.device) for g in masked_gs]
                    if feature_mask is not None:
                        if self.support_multi_features and self.use_meta:
                            features = [features * i.to(self.device_string).view(-1, 1) for i in
                                        feature_mask]
                        else:
                            feature_mask_device = feature_mask.to(self.device_string)
                            features = features * feature_mask_device.view(-1, 1)
                    return gs, features

                return handle_fn
        elif self.abstract_explainer_node_name == "SubgraphXCore":
            def used_fn(masked_gs=None, feature_mask=None):
                if masked_gs is None:
                    masked_gs, features = self.extract_neighbors_input()
                else:
                    masked_gs = [i.to(self.device_string) for i in masked_gs]
                if feature_mask is None:
                    feature_mask = self.feature_mask_for_output_out if not self.move2cpu else \
                        self.feature_mask_for_output_out.to(self.device_string)
                else:
                    feature_mask = [i.to(self.device_string) for i in feature_mask]
                _, features = self.extract_neighbors_input()
                feature_mask = [
                    features * i.view(-1, 1) for i in feature_mask
                ]

                return lambda model: (masked_gs, feature_mask)
        else:
            raise NotImplementedError(f"Abstract explainer node {self.abstract_explainer_node_name} is not supported.")

        self.get_custom_input_handle_fn = used_fn
