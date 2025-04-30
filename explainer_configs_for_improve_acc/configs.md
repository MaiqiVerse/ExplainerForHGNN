### Example GNNExplainer Configuration:

```json
{
  "record_metrics": [
    "mask_density"
  ],
  "extract_neighbors": true,
  "n_hop": 3,
  "init_strategy_for_edge": "normal",
  "use_mask_bias": false,
  "init_strategy_for_feature": "const",
  "edge_mask_threshold": 0.5,
  "feature_mask_threshold": 0.5,
  "opt_scheduler": null,
  "opt": "adam",
  "edge_mask_activation": "sigmoid",
  "epochs": 100,
  "record": true,
  "record_step": 1,
  "feature_mask_use_sigmoid": true,
  "coff_normal": 1.0,
  "coff_edge_size": 0.005,
  "coff_feature_size": 1.0,
  "coff_edge_ent": 1.0,
  "coff_feature_ent": 0.1,
  "coff_laplacian": 1.0,
  "mask_features": true,
  "feature_mask_marginalize": false,
  "opt_lr": 0.1,
  "opt_wd": 0.0,
  "eval_metrics": [
    "Macro-F1",
    "Micro-F1"
  ],
  "summary_path": "gnn_explainer_summary.json"
}
```

### GNNExplainer Configuration Keys (with descriptions):

1. **Explainer Parameters:**

   - **`extract_neighbors`**: (boolean)
     If `true`, the explainer will extract neighbors of the node to be explained, otherwise, the whole graph will be used.
   - **`n_hop`**: (integer)
     Number of hops to extract neighbors for explanation. Only used when `extract_neighbors` is `true`. The value is usually determined by the number of graph convolution layers in the model.
   - **`mask_features`**: (boolean)
     If `true`, feature masking is applied during the explanation process.
   - It should be set to `false` in this improve acc configuration.
   - **`feature_mask_marginalize`**: (boolean)
     If `true`, the explainer will marginalize the feature mask, adding noise to masked features.
2. **Mask Initialization:**

   - **`init_strategy_for_edge`**: (string)
     Initialization strategy for edge masks. Available options: `"normal"` for normal distribution initialization, or `"constant"` for constant value initialization.
   - **`init_strategy_for_feature`**: (string)
     Initialization strategy for feature masks. Available options: `"normal"` or `"const"`.
   - **`use_mask_bias`**: (boolean)
     If `true`, the explainer will apply a bias term to the mask values.
   - **`init_const`**: (float or integer)
     Constant value for edge mask initialization.
     
3. **Mask Activation and Thresholding:**

   - **`edge_mask_activation`**: (string)
     Activation function for edge masks. Options are `"sigmoid"` or `"relu"`.
   - **`edge_mask_threshold`**: (float)
     Threshold value for applying edge masks. Mask values below the threshold are removed.
   - **`feature_mask_threshold`**: (float)
     Threshold value for feature masks.
4. **Training Parameters:**

   - **`epochs`**: (integer)
     Number of epochs to train the explainer.
   - **`record`**: (boolean)
     If `true`, the metrics during training will be recorded.
   - **`record_step`**: (integer)
     Interval (in epochs) to record metrics during training. Only used if `record` is `true`.
   - **`device`**: (integer)
     Device to use for training. If set to `null`, the CPU will be used.
5. **Optimization Parameters:**

   - **`opt`**: (string)
     Specifies the optimizer to use. Options are `"adam"`, `"sgd"`, `"rmsprop"`, or `"adagrad"`.
   - **`opt_lr`**: (float)
     Learning rate for the optimizer.
   - **`opt_wd`**: (float)
     Weight decay for the optimizer.
   - **`opt_scheduler`**: (string, optional)
     Scheduler for learning rate adjustment. Options are `"null"` (no scheduler), `"step"` (step decay), or `"cos"` (cosine annealing).
6. **Coefficients (Regularization Terms):**

   - **`coff_normal`**: (float)
     Coefficient for the standard loss term (e.g., cross-entropy loss).
   - **`coff_edge_size`**: (float)
     Coefficient for the edge mask size loss. Penalizes large edge masks.
   - **`coff_feature_size`**: (float)
     Coefficient for the feature mask size loss. Penalizes large feature masks.
   - **`coff_edge_ent`**: (float)
     Coefficient for the edge mask entropy loss. Promotes sparse and smooth edge masks.
   - **`coff_feature_ent`**: (float)
     Coefficient for the feature mask entropy loss.
   - **`coff_laplacian`**: (float)
     Coefficient for the Laplacian loss, which encourages connectedness in the graph structure.
7. **Evaluation and Metrics:**

   - **`eval_metrics`**: (list of strings) Metrics to evaluate the explanation. In current improve acc setting, only Macro-F1 and Micro-F1 is supported. Available options include:
     - `"Macro-F1"`: Macro-F1 score.
     - `"Micro-F1"`: Micro-F1 score.
   - **`record_metrics`**: (list of strings)
     Metrics to record during training. Example: `["mask_density"]` tracks the density of the mask.
8. **File Paths:**

   - **`summary_path`**: (string)
     Path to save the explanation summary. If set to `null`, the summary will not be saved.
