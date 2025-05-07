import json
import os

__all__ = [
    "load_model",
    "HAN",
    # "MAGNN",
    # "HGT",
    # "RGCN",
    "BaseModel",
    "HAN_GCN"
]

from .han import HAN
# from .gcn import GCN
# from .gat import GAT
# from .magnn import MAGNN
# from .hgt import HGT
# from .rgcn import RGCN
from .model import BaseModel
from .han_gcn import HAN_GCN


def load_model(model_name, dataset, model_config=None, is_load_model=False, device=None):
    # load config first
    if model_config is not None:
        with open(model_config, "r") as f:
            config = json.load(f)
    elif os.path.exists(f"./model_configs/{model_name}.json"):
        with open(f"./model_configs/{model_name}.json", "r") as f:
            config = json.load(f)
    else:
        with open(f"./model_configs/{model_name}_{dataset.dataset_name}.json",
                  "r") as f:
            config = json.load(f)

    if device is not None:
        config["device"] = device

    # enable reusing saved model
    if is_load_model and config.get("weight_path", None):
        model_path = config["weight_path"]
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model {model_path} not found")
        else:
            return globals()[model_name].load(model_path, dataset)

    if model_name == "HAN":
        return HAN(config, dataset)
    # elif model_name == "MAGNN":
    #     return MAGNN(config, dataset)
    # elif model_name == "HGT":
    #     return HGT(config, dataset)
    # elif model_name == "RGCN":
    #     return RGCN(config, dataset)
    elif model_name == "HAN_GCN":
        return HAN_GCN(config, dataset)
    else:
        return BaseModel(config, dataset)
