import copy
import torch
import torch.nn as nn
import sklearn.metrics as metrics


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









