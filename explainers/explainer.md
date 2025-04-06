# Standard Explainer

Currently, we only support node classification tasks practically. It supports graph classification in design level.
We are not sure how it supports link prediction in future design.

Here for a standard explainer, we utilize two key components:

1. ExplainerCore: It focuses on single node explanation.
2. Explainer: It can provide the whole explanation for a model in a given dataset.

## Base

### ExplainerCore

`explainers.explainer.ExplainerCore` is the base class for all ExplainerCores.

It defines the following methods:

- `__init__`: Initialize the ExplainerCore with a model and a dataset.
    - requires a config dict as input
- `explain`: Generate explanations for a given node.
    - requires a model as input
    - other parameters are optional
- `node_level_explain`: Generate explanations for a given node.
    - requires a model as input
    - requires a node id as input
    - other parameters are optional
- `graph_level_explain`: Generate explanations for a given graph (graph classification).
    - requires a model as input
    - other parameters are optional
- `construct_explanation`: Construct an explanation for a given node.
- `init_params`: Initialize parameters for the ExplainerCore. Note that only if you implement a training-based
  explainer, you need to implement this method.
- `init_params_node_level`: Initialize parameters for the node-level explainer. Note that only if you implement a training-based
  explainer, you need to implement this method.
- `init_params_graph_level`: Initialize parameters for the graph-level explainer. Note that only if you implement a training-based
  explainer, you need to implement this method.
- `extract_neighbors_input`: Extract neighbors from the input node. Consider that most explainers need
  neighbors from the input node to get a subgraph to reduce the computation cost.
- `visualize`: Visualize the explanation. Note that all existing explainers do not implement this method.
- `get_required_fit_params`: Get the required parameters for training process, so if this explainer is a training-based explainer,
  you need to implement this method.
- `fit`: Fit the explainer. Usually are the key logic of the explainer. 
- `fit_node_level`: Fit the explainer for node-level explanation. Usually are the key logic of the explainer.
- `fit_graph_level`: Fit the explainer for graph-level explanation. Usually are the key logic of the explainer.
- `get_loss`: Get the loss for the explainer. Only if you implement a training-based explainer, you need to implement this method.
- `get_loss_node_level`: Get the loss for the node-level explainer. Only if you implement a training-based explainer, you need to implement this method.
- `get_loss_graph_level`: Get the loss for the graph-level explainer. Only if you implement a training-based explainer, you need to implement this method.
- `get_input_handle_fn`: Get the input handle function. The model will need the function which can return the modified input to generate the customized output.
- `get_input_handle_fn_node_level`: Get the input handle function for node-level explanation. The model will need the function which can return the modified input to generate the customized output.
- `get_input_handle_fn_graph_level`: Get the input handle function for graph-level explanation. The model will need the function which can return the modified input to generate the customized output.
- `forward`: The forward propagation function. Only if you implement a training-based explainer, you must implement this method.
- `forward_node_level`: The forward propagation function for node-level explanation.
- `forward_graph_level`: The forward propagation function for graph-level explanation.
- `build_optimizer`: Build the optimizer for the explainer. Only if you implement a training-based explainer, you need to implement this method.
- `build_scheduler`: Build the scheduler for the explainer. Only if you implement a training-based explainer, you need to implement this method.
- `to`: Move the explainer to a specific device. Most of the time, you do not need to implement this method.

It also defines the following properties:
- `config`: The config dict for the explainer. Setting in `__init__` method.
- `model`: The model for the explainer. Initialized in `__init__` method. Setting in `explain` method. 
- `meterics`: The metrics for the explainer. Usually from the `eval_metrics` in the config. Setting in `__init__` method.
- `device`: The device for the explainer. Initialized in `__init__` method.
- `registered_modules_and_params`: The registered modules and parameters for the explainer. Initialized in `__init__` method.
- `original_explainer`: If it is an original explainer or meta-path-based explainer. Duplicated. Initialized in `__init__` method.
- `edge_mask`: The edge mask for the explainer. Initialized in `init_params_node_level` method.
- `feature_mask`: The feature mask for the explainer. Initialized in `init_params_node_level` method.
- `neighbor_input`: The neighbor input for the explainer. Initialized in `extract_neighbors_input` method.
    - `gs`: The graphs (from a heterogeneous graph). Initialized in `extract_neighbors_input` method.
    - `feature`: The node feature. Initialized in `extract_neighbors_input` method.
    - `n_hop`: The number of hops of the neighbors. Initialized in `extract_neighbors_input` method. Usually from the config.
    - `used_nodes`: The used nodes. Initialized in `extract_neighbors_input` method.
    - `recovery_dict`: The recovery dict. Recover the mapped node id to the original node id. Initialized in `extract_neighbors_input` method.
- `edge_mask_for_output`: Requires for the output to generate evaluation metrics. A property function.
- `feature_mask_for_output`: Requires for the output to generate evaluation metrics. A property function.
- `get_custom_input_handle_fn`: Get the custom input handle function. It must be defined to generate evaluation metrics.
Because when calculating the metrics, the masks will be further processed to get the hard mask.
- `device_string`: We usually use integer to represent the gpu device. To represent as "cuda:0" or "cpu",
  we need to convert it to string. A property function.


## `Explainer`

`explainers.explainer.Explainer` is the base class for all explainers.
It defines the following methods:
- `__init__`: Initialize the Explainer with a model and a dataset.
    - requires a config dict as input
- `explain`: Generate explanations for a given node.
    - requires a model as input
    - other parameters are optional
- `node_level_explain`: Generate explanations for a given node.
    - other parameters are optional
- `graph_level_explain`: Generate explanations for a given graph (graph classification).
    - other parameters are optional
- `construct_explanation`: Construct the output explanation.
- `evaluate`: Evaluate the explanation.
- `visualize`: Visualize the explanation. Note that all existing explainers do not implement this method.
- `get_summary`: Get the summary of the evaluation.
- `save_summary`: Save the summary to a file.
- `to`: Move the explainer to a specific device. Most of the time, you do not need to implement this method.
- `save_explanation`: Save the explanation to a file.


## Normally Implementation

### ExplainerCore

#### Method: `__init__`

Get `record_metrics` from the config.

```python
def __init__(self, config):
    super().__init__(config)
    self.record_metrics = self.config.get('record_metrics', None)
    if not self.record_metrics:
        self.record_metrics = ['mask_density']
```

#### Method: `explain`

```python
def explain(self, model, **kwargs):
    self.model = model
    self.model.eval()

    if self.model.dataset.single_graph:
        self.node_id = kwargs.get('node_id', None)

    self.init_params()

    if self.model.dataset.single_graph:
        if self.node_id is None:
            raise ValueError('node_id is required for node-level explanation')
        return self.node_level_explain()
    else:
        return self.graph_level_explain()
```

#### Method: `node_level_explain`

```python
def node_level_explain(self):
    self.fit()
    return self.construct_explanation()
```

#### Method: `graph_level_explain`

```python
def graph_level_explain(self):
    pass
```

#### Method: `construct_explanation`

```python
from .prepare_explanation_for_node_scores import standard_explanation
from .prepare_explanation_for_node_dataset_scores import \
    prepare_explanation_fn_for_node_dataset_scores
from .explanation import NodeExplanation


def construct_explanation(self):
    explanation = NodeExplanation()
    explanation = standard_explanation(explanation, self)
    for metric in self.config['eval_metrics']:
        prepare_explanation_fn_for_node_dataset_scores[metric](explanation, self)
    self.explanation = explanation
    return explanation
```

#### Method: `init_params`

```python
def init_params(self):
    if self.model.dataset.single_graph:
        self.init_params_node_level()
    else:
        self.init_params_graph_level()
    self.registered_modules_and_params = {
        str(index): i for index, i in enumerate(self.get_required_fit_params())
    }
    self.to(self.device_string)
```

#### Method: `init_params_node_level`

```python
def init_params_node_level(self):
    pass
```

For those training-based explainers, you could see GNNExplainerMetaCore as an example.

#### Method: `init_params_graph_level`

```python
def init_params_graph_level(self):
    pass
```

#### Method: `extract_neighbors_input`

```python
import torch


def extract_neighbors_input(self, node_id):
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
          # !TODO: Test it in the future, and then expand it to other algorithms
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
```

#### Method: `visualize`

```python
def visualize(self, explanation):
    pass
```

#### Method: `get_required_fit_params`

Return a list of required parameters for training process.

#### Method: `fit`

```python
def fit(self):
    if self.model.dataset.single_graph:
        self.fit_node_level()
    else:
        self.fit_graph_level()
```


#### Method: `fit_node_level`

Implement your own logic here.

#### Method: `fit_graph_level`

```python
def fit_graph_level(self):
    pass
```

#### Method: `get_loss`

```python
def get_loss(self, output, mask=None):
    if self.model.dataset.single_graph:
        return self.get_loss_node_level(output, mask)
    else:
        return self.get_loss_graph_level(output, mask)
```

#### Method: `get_loss_node_level`

Return the loss for node-level explanation.

#### Method: `get_loss_graph_level`

```python
def get_loss_graph_level(self, output, mask=None):
    pass
```

#### Method: `get_input_handle_fn`

```python
def get_input_handle_fn(self):
    if self.model.dataset.single_graph:
        return self.get_input_handle_fn_node_level()
    else:
        return self.get_input_handle_fn_graph_level()
```

#### Method: `get_input_handle_fn_node_level`

```python
def get_input_handle_fn_node_level(self):
    def handle_fn(model):
        gs, features = self.extract_neighbors_input()
        
        # you can add your own logic here

    return handle_fn
```

#### Method: `get_input_handle_fn_graph_level`

```python
def get_input_handle_fn_graph_level(self):
    pass
```

#### Method: `forward`

```python
def forward(self):
    if self.model.dataset.single_graph:
        return self.forward_node_level()
    else:
        return self.forward_graph_level()
```

#### Method: `forward_node_level`

Implement your own logic here.

#### Method: `forward_graph_level`

```python
def forward_graph_level(self):
    pass
```

#### Method: `build_optimizer`

Pass

#### Method: `build_scheduler`

Pass

#### Method: `to`

Pass

#### Method: `save_explanation`

Pass

#### Method: `save_summary`

Pass

#### Method: `get_custom_input_handle_fn`

```python
def get_custom_input_handle_fn(self, masked_gs=None, feature_mask=None):
    """
    Get the custom input handle function for the model.
    :return:
    """

    def handle_fn(model):
        if model is None:
            model = self.model
        gs, features = self.extract_neighbors_input()
        if masked_gs is not None:
            gs = [i.to(self.device_string) for i in masked_gs]
        if feature_mask is not None:
            feature_mask_device = feature_mask.to(self.device_string)
            features = features * feature_mask_device
        return gs, features

    return handle_fn
```

### `Explainer`

#### Method: `__init__`

```python
def __init__(self, config):
    super().__init__(config)
```

#### Method: `explain`

```python
def explain(self, model, **kwargs):
    self.model = model

    if self.model.dataset.single_graph:
        return self.node_level_explain(**kwargs)
    else:
        return self.graph_level_explain(**kwargs)
```

#### Method: `node_level_explain`

```python
def node_level_explain(self, **kwargs):

    result = []
    test_labels = self.model.dataset.labels[2]

    if kwargs.get('max_nodes', None) is not None \
        and kwargs.get('max_nodes') < len(test_labels):
        test_labels = test_labels[:kwargs.get('max_nodes')]

    for idx, label in test_labels:
        explain_node = ExplainerCore(self.config)
        explain_node.to(self.device)
        explanation = explain_node.explain(self.model,
                                           node_id=idx)
        result.append(explanation)

    # result = NodeExplanationCombination(node_explanations=result)

    result = self.construct_explanation(result)

    self.result = result

    self.evaluate()

    self.save_summary()

    return self.eval_result
```

#### Method: `graph_level_explain`

```python
def graph_level_explain(self, **kwargs):
    pass
```

#### Method: `construct_explanation`

```python
from .explanation import NodeExplanationCombination


def construct_explanation(self, result):
    result = NodeExplanationCombination(node_explanations=result)
    if self.config.get('control_data', None) is not None:
        result.control_data = self.config['control_data']

    return result
```

#### Method: `evaluate`

```python
from .prepare_combined_explanation_for_node_dataset_scores import \
    prepare_combined_explanation_fn_for_node_dataset_scores
from .node_dataset_scores import node_dataset_scores


def evaluate(self):
    eval_result = {}
    if self.config.get('eval_metrics', None) is not None:
        for metric in self.config['eval_metrics']:
            self.result = prepare_combined_explanation_fn_for_node_dataset_scores[
                metric](self.result, self)
            eval_result[metric] = node_dataset_scores[metric](self.result)

    self.eval_result = eval_result
    return eval_result
```

#### Method: `get_summary`

```python
def get_summary(self):
    return self.eval_result
```

#### Method: `save_summary`

```python
def save_summary(self):
    if self.config.get('summary_path', None) is not None:
        import os
        os.makedirs(os.path.dirname(self.config['summary_path']),
                    exist_ok=True)
        import json
        with open(self.config['summary_path'], 'w') as f:
            json.dump(self.eval_result, f)
```


#### Method: `visualize`

Pass

#### Method: `save_explanation`

```python
def save_explanation(self, **kwargs):
    if self.config.get('explanation_path', None) is not None:
        import os
        os.makedirs(self.config['explanation_path'],
                    exist_ok=True)
        self.result.save(self.config['explanation_path'], **kwargs)
```


#### Method: `to`

Pass

#### Method: `explain_selected_nodes`

```python
def explain_selected_nodes(self, model, selected_nodes):
    self.model = model
    result = []
    test_labels = self.model.dataset.labels[2]
    for idx, label in test_labels:
        if idx in selected_nodes:
            explain_node = ExplainerCore(self.config)
            explain_node.to(self.device)
            explanation = explain_node.explain(self.model,
                                               node_id=idx)
            result.append(explanation)

    result = self.construct_explanation(result)

    self.result = result

    self.evaluate()

    self.save_summary()
```

## Conclusion

Hope this simple document can help you to implement your own explainer.