import json

__all__ = [
    "load_explainer",
    "GradExplainerMeta",
    "GradExplainer",
    "GradExplainerOriginal",
    "GradExplainerCore",
    "GradCAM",
    # "SAExplainer",
    # "GOAtExplainer",
    # "SubgraphXExplainer",
    "GNNExplainerMeta",
    "GNNExplainerOriginal",
    "GNNExplainerMetaCore",
    "GNNExplainerOriginalCore",
    # "PGExplainer",
    # "CGEExplainer",
    "Explainer",
    "ExplainerCore",
    "load_default_config",
    "NodeExplanation",
    "NodeExplanationCombination",
    "HENCEX",
    "HENCEXCore",
    "RandomExplainer",
    "RandomExplainerCore",
    "RandomEdgeMaskExplainer",
    "RandomFeatureMaskExplainer",
    "RandomNodeMaskExplainer",
    "RandomEdgeAndFeatureMaskExplainer",
    "RandomEdgeAndNodeMaskExplainer",
]

from .grad import GradExplainerMeta, GradExplainer, GradExplainerOriginal, \
    GradExplainerCore
# from .saliency import SAExplainer
# from .goat import GOAtExplainer
# from .subgraphx import SubgraphXExplainer
from .gnnexplainer import GNNExplainerMeta, GNNExplainerOriginal, GNNExplainerMetaCore, \
                        GNNExplainerOriginalCore
# from .pgexplainer import PGExplainer
# from .cge import CGEExplainer
from .explainer import Explainer, ExplainerCore
from .explanation import NodeExplanation, NodeExplanationCombination
from .hencex import HENCEX, HENCEXCore
from .random import RandomExplainer, RandomExplainerCore, RandomEdgeMaskExplainer, \
    RandomFeatureMaskExplainer, RandomNodeMaskExplainer, RandomEdgeAndFeatureMaskExplainer, \
    RandomEdgeAndNodeMaskExplainer
from .cebased import CEBased, CEBasedCore
from .gradcam import GradCAM, GradCAMCore, GradCAMMeta, GradCAMOriginal
import os


def load_default_config(explainer_name, model_name, dataset_name):
    if os.path.exists(
        f"./explainer_configs/{explainer_name}_{model_name}_{dataset_name}.json"):
        with open(
            f"./explainer_configs/{explainer_name}_{model_name}_{dataset_name}.json",
            "r") as f:
            config = json.load(f)
    else:
        with open(f"./explainer_configs/{explainer_name}.json", "r") as f:
            config = json.load(f)
        for key in config.keys():
            if key.endswith("_path"):
                config[key] = config[key].format(model_name=model_name,
                                                 dataset_name=dataset_name,
                                                 explainer_name=explainer_name)
    return config


def load_explainer(explainer_name, model_name, dataset_name, explainer_config=None):
    # load config first
    if explainer_config is not None:
        with open(explainer_config, "r") as f:
            config = json.load(f)
    else:
        config = load_default_config(explainer_name, model_name, dataset_name)

    if explainer_name == "GNNExplainerMeta":
        return GNNExplainerMeta(config)
    elif explainer_name == "GNNExplainerOriginal":
        return GNNExplainerOriginal(config)
    elif explainer_name == "GradExplainerMeta":
        return GradExplainerMeta(config)
    elif explainer_name == "GradExplainer":
        return GradExplainer(config)
    elif explainer_name == "GradExplainerOriginal":
        return GradExplainerOriginal(config)
    elif explainer_name == "HENCEX":
        return HENCEX(config)
    elif explainer_name == "RandomExplainer":
        return RandomExplainer(config)
    elif explainer_name == "RandomEdgeMaskExplainer":
        return RandomEdgeMaskExplainer(config)
    elif explainer_name == "RandomFeatureMaskExplainer":
        return RandomFeatureMaskExplainer(config)
    elif explainer_name == "RandomNodeMaskExplainer":
        return RandomNodeMaskExplainer(config)
    elif explainer_name == "RandomEdgeAndFeatureMaskExplainer":
        return RandomEdgeAndFeatureMaskExplainer(config)
    elif explainer_name == "RandomEdgeAndNodeMaskExplainer":
        return RandomEdgeAndNodeMaskExplainer(config)
    elif explainer_name == "CEBased":
        return CEBased(config)
    elif explainer_name == "GradCAM":
        return GradCAM(config)
    elif explainer_name == "GradCAMMeta":
        return GradCAMMeta(config)
    elif explainer_name == "GradCAMOriginal":
        return GradCAMOriginal(config)
    # elif explainer_name == "SAExplainer":
    #     return SAExplainer(config)
    # elif explainer_name == "GOAtExplainer":
    #     return GOAtExplainer(config)
    # elif explainer_name == "SubgraphXExplainer":
    #     return SubgraphXExplainer(config)
    # elif explainer_name == "GNNExplainer":
    #     return GNNExplainer(config)
    # elif explainer_name == "PGExplainer":
    #     return PGExplainer(config)
    # elif explainer_name == "CGEExplainer":
    #     return CGEExplainer(config)
    else:
        print(f"Warning: Explainer {explainer_name} not found.")
        return Explainer(config)
