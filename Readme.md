# Do We Really Need to Access the meta-path in Heterogeneous Information Networks: An Explainable Graph Neural Network Approach

## Introduction

This repository implements several explainable graph neural network models, and applies them to heterogeneous GNNs.
We do fair comparison between different models and evaluate their performance on several benchmark datasets.

If you find this repository useful, please star it. We will provide the citation information soon.

## Requirements

- Python 3.10
- Pytorch (developed with 2.3.0)
- Cuda 11 or higher (if you want to use GPU. Not sure about the lower version of Cuda)
- Numpy

Note: Pytorch seems to do a lot of changes in the sparse tensor (Compared to the previous version), and give
a better support for the sparse tensor, which we rely on. So the elder version of Pytorch may not work with this code.

Note: Currently, we not rely on other common libraries like `torch_geometric`, `dgl`, etc. We hope to make the code more
compatible with different environments. But if it is tough to implement the model without these libraries in the future,
we will use them.

## Performance

Example results for Some of Explainers on the HAN model with GCN on the ACM dataset:

### Results of explanation models on HAN-GCN model for ACM dataset (%)

| Model                           | 1 - Fidelity\_-      | Fidelity\_+         | Characterization Score | Macro-F1          | Micro-F1          |
|--------------------------------|----------------------|---------------------|-------------------------|-------------------|-------------------|
| GNNExplainer (Meta-Path)       | 90.15 ± 2.99         | 45.50 ± 6.62        | 60.20 ± 6.10            | 82.77 ± 3.29      | 82.50 ± 3.28      |
| Grad (Meta-Path)               | 96.40 ± 1.50         | 35.80 ± 2.42        | 52.16 ± 2.57            | 88.22 ± 1.43      | 88.05 ± 1.43      |


## Dataset

### GTN Datasets

We use GTN datasets, including ACM, DBLP, IMDB, available at
[GTN](https://github.com/seongjunyun/Graph_Transformer_Networks).

### Create edge directions

We notice that many heterogeneous models require the edge direction. We provide an open API to create the edge
direction:

```python
from utils import edge_direction_creation

edge_direction_creation.process_node_classification_dataset_to_edge_directions(dataset_path={"Your dataset"},
                                                                               label_path={"Your labels.pkl file"})
```

## Run the code

To run the code, you can use the following command:

```bash
python main.py --dataset {dataset_name} --model {model_name} --explainer {explainer_name} --random_seed {seed}
```

You can do some customization by
changing `./dataset_configs/{dataset_name}.json`, `./model_configs/{model_name}_{dataset_name}.json`
and `./explainer_configs/{explainer_name}_{model_name}_{dataset_name}.json`.

Note: Although we recommend using `./explainer_configs/{explainer_name}_{model_name}_{dataset_name}.json` to set
the details for different models and datasets, you can just use `./explainer_configs/{explainer_name}.json` to set the
new model and dataset. You can just provide the `./explainer_configs/{explainer_name}.json` if you design a new
explainer.

More details can be found in the `python main.py -h`.

### Multi-run

You can run multiple experiments by using the following command:

```bash
python main_multiple_times.py --dataset {dataset_name} --model {model_name} --explainer {explainer_name} --random_seed {seeds}
```

Seeds should be a list of integers, e.g., `--random_seed 1 2 3 4 5`.

We also offer some customization settings in `python main_multiple_times.py -h`. For example, you can find setting that
allows you to run the experiments in different model settings, and then you do not need to give the random seeds.

## Explainer

Now Dataset available: ACM, DBLP, IMDB

Now model available: HAN (with GAT), HAN (with GCN)

Now explainer available: GNNExplainer, Grad, HENCEX (only one version)

Officially, you can use `GNNExplainerMeta`, `GradExplainerMeta`, `HENCEX` to
run the experiments.

Some explainers (i.e. GOAt) require more general interface in model than the default design. We do not ensure that all
explainers can work with all models. If you want to use a new model, you may need additional modification in the model.

More importantly, explainers like GOAt have strict limitation on such as layer type, and furthermore, intrusive design
in explainers into the model makes them hard to be generalized. We will try to make the code more general, but it is
limited by the design of the explainer itself. You still can consider to modify the explainer to expand the model
support, but these modifications may not be easy, and may out of the scope of their original design, especially for
complicated attention mechanism in the model.

### Plan:

#### Homogeneous Graph

PGExplainer

```bibtex
@article{luo2020parameterized,
    title = {Parameterized Explainer for Graph Neural Network},
    author = {Luo, Dongsheng and Cheng, Wei and Xu, Dongkuan and Yu, Wenchao and Zong, Bo and Chen, Haifeng and Zhang, Xiang},
    journal = {Advances in Neural Information Processing Systems},
    volume = {33},
    year = {2020}
}
```

GNNExplainer

```bibtex
@article{Ying2019GNNExplainerGE,
    title = {GNNExplainer: Generating Explanations for Graph Neural Networks},
    author = {Rex Ying and Dylan Bourgeois and Jiaxuan You and Marinka Zitnik and Jure Leskovec},
    journal = {Advances in neural information processing systems},
    year = {2019},
    volume = {32},
    pages = {
    9240-9251
        },
    url = {https://api.semanticscholar.org/CorpusID:202572927}
}
```

SubgraphX

```bibtex
@inproceedings{Yuan2021OnEO,
    title = {On Explainability of Graph Neural Networks via Subgraph Explorations},
    author = {Hao Yuan and Haiyang Yu and Jie Wang and Kang Li and Shuiwang Ji},
    booktitle = {International Conference on Machine Learning},
    year = {2021},
    url = {https://api.semanticscholar.org/CorpusID:231861768}
}
```

SA

```bibtex
@inproceedings{baldassarre2019explainability,
    title = {Explainability techniques for graph convolutional networks},
    author = {Baldassarre, Federico and Azizpour, Hossein},
    booktitle = {Proceedings of the ICML 2019 Workshop on Learning and Reasoning with Graph-Structured Representations},
    year = {2019},
    note = {arXiv preprint arXiv:1905.13686}
}
```

GOAt

```bibtex
@inproceedings{Lu2024GOAtEG,
    title = {GOAt: Explaining Graph Neural Networks via Graph Output Attribution},
    author = {Shengyao Lu and Keith G. Mills and Jiao He and Bang Liu and Di Niu},
    booktitle = {Proceedings of the International Conference on Learning Representations (ICLR)},
    year = {2024},
    url = {https://api.semanticscholar.org/CorpusID:267301280},
    note = {arXiv preprint arXiv:2401.14578}
}

```

Grad

```bibtex
@article{Selvaraju2016GradCAMVE,
    title = {Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization},
    author = {Ramprasaath R. Selvaraju and Abhishek Das and Ramakrishna Vedantam and Michael Cogswell and Devi Parikh and Dhruv Batra},
    journal = {International Journal of Computer Vision},
    year = {2016},
    volume = {128},
    pages = {336 - 359},
    url = {https://api.semanticscholar.org/CorpusID:15019293}
}
```

The applicable version for GNN is:

```bibtex
@INPROCEEDINGS{8954227,
    author = {Pope, Phillip E. and Kolouri, Soheil and Rostami, Mohammad and Martin, Charles E. and Hoffmann, Heiko},
    booktitle = {2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    title = {Explainability Methods for Graph Convolutional Neural Networks},
    year = {2019},
    volume = {},
    number = {},
    pages = {10764-10773},
    keywords = {Deep Learning;Deep Learning},
    doi = {10.1109/CVPR.2019.01103} }
```

#### Heterogeneous Graph

PaGE-Link (Only for Link Prediction)

```bibtex
@article{Zhang2023PaGELinkPG,
  title={PaGE-Link: Path-based Graph Neural Network Explanation for Heterogeneous Link Prediction},
  author={Shichang Zhang and Jiani Zhang and Xiang Song and Soji Adeshina and Da Zheng and Christos Faloutsos and Yizhou Sun},
  journal={Proceedings of the ACM Web Conference 2023},
  year={2023},
  url={https://api.semanticscholar.org/CorpusID:257205930}
}
```

CE-based

```bibtex
@article{Khler2024UtilizingDL,
  title={Utilizing Description Logics for Global Explanations of Heterogeneous Graph Neural Networks},
  author={Dominik K{\"o}hler and Stefan Heindorf},
  journal={ArXiv},
  year={2024},
  volume={abs/2405.12654},
  url={https://api.semanticscholar.org/CorpusID:269930057}
}
```

CF-HGExplainer

```bibtex
@article{Yang2023CounterfactualLO,
  title={Counterfactual Learning on Heterogeneous Graphs with Greedy Perturbation},
  author={Qiang Yang and Changsheng Ma and Qiannan Zhang and Xin Gao and Chuxu Zhang and Xiangliang Zhang},
  journal={Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2023},
  url={https://api.semanticscholar.org/CorpusID:260499632}
}
```

Will be implemented if the authors really provide the code. Currently, the author repository is empty.

```github
https://github.com/gitdevqiang/CF-HGExplainer
```

HENCE-X

```bibtex
@article{Lv2023HENCEXTH,
  title={HENCE-X: Toward Heterogeneity-agnostic Multi-level Explainability for Deep Graph Networks},
  author={Gengsi Lv and C. Zhang and Lei Chen},
  journal={Proc. VLDB Endow.},
  year={2023},
  volume={16},
  pages={2990-3003},
  url={https://api.semanticscholar.org/CorpusID:261197657}
}
```
