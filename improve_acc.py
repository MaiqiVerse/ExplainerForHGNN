"""
This file is trying to use explanation model to denoise the neighborhood of target node, in order to improve model
accuracy.
This is quick displaying mode, we will not keep many customized args to avoid complexity.
"""
import torch
import numpy as np
import random
import os
import json

from explainers import load_explainer
from datasets import load_dataset
from models import load_model


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Improve model accuracy by denoising the neighborhood of target node')
    parser.add_argument('--model', type=str, default='HAN_GCN',
                        help='The model to be used for explanation')
    parser.add_argument('--dataset', type=str, default='./data/ACM',
                        help='Path to dataset, the folder name should be the same as the dataset name')
    parser.add_argument('--explainer', type=str, default='GNNExplainerMeta',
                        help='The explainer to be used for explanation')
    parser.add_argument('--device', type=int, default=0,
                        help='GPU device id')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()
    return args


def set_seed(seed, ensure_reproducibility=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    torch.use_deterministic_algorithms(True)
    import os
    # because of the suggestion from
    # https://pytorch.org/docs/stable/notes/randomness.html
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if ensure_reproducibility:
        torch.backends.cudnn.benchmark = False


def train_model(model_name, dataset_path, device, dataset_config=None,
                model_config=None
                ):
    dataset = load_dataset(dataset_path, dataset_config)
    model = load_model(model_name, dataset, model_config)
    model.to(device)

    # Train model
    print("Training model...")
    model.train()
    model.fit()
    summary = model.get_summary()

    # print summary
    print("Model Summary:")
    print("----------------")
    for key, value in summary.items():
        print(f"{key}: {value}")

    return model


def explain(model, explainer_name, device, explainer_config=None
            ):
    explainer = load_explainer(explainer_name, model.__class__.__name__,
                               model.dataset.__class__.__name__,
                               explainer_config)
    explainer.to(device)
    result = explainer.explain(model)
    return explainer


def main():
    args = get_args()
    set_seed(args.random_seed, True)

    # Set device
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # Load model
    model = train_model(args.model, args.dataset, device)

    # Get the explanation for the target node
    explainer_config = prepare_improve_acc_explainer_config(args.explainer, args.model, args.dataset)
    explainer = explain(model, args.explainer, args.device, explainer_config)

    # Get the improvement summary
    improvement_summary = get_improvement_summary(explainer)

    # Print the improvement summary
    print("Improvement Summary:")
    print("----------------")
    for key, value in improvement_summary.items():
        print(f"{key}: {value}")

    # Save the summary
    save_summary(improvement_summary, explainer)


def prepare_improve_acc_explainer_config(explainer_name, model_name, dataset_name):
    default_folder = "./explainer_configs_for_improve_acc/"
    if not os.path.exists(default_folder):
        os.makedirs(default_folder)

    # test {explainer_name}_{model_name}_{dataset_name}.json is exist
    # if not, test {explainer_name}_{model_name}.json is exist
    # if not, test {explainer_name}.json is exist
    if os.path.exists(
            f"{default_folder}{explainer_name}_{model_name}_{dataset_name}.json"):
        return os.path.join(default_folder,
                            f"{explainer_name}_{model_name}_{dataset_name}.json")
    elif os.path.exists(
            f"{default_folder}{explainer_name}_{model_name}.json"):
        return os.path.join(default_folder,
                            f"{explainer_name}_{model_name}.json")
    elif os.path.exists(
            f"{default_folder}{explainer_name}.json"):
        return os.path.join(default_folder, f"{explainer_name}.json")
    else:
        raise ValueError(
            f"Explainer config file for {explainer_name} not found in {default_folder}")


def get_improvement_summary(explainer):
    """
    Get the improvement summary from the explainer
    :param explainer: an Explainer object
    :return: a dictionary containing the improvement summary
    """
    summary = explainer.get_summary()
    origin_summary = explainer.model.get_summary()
    improved = "Macro-F1 ({}) Micro-F1 ({})".format(
        summary["Macro-F1"], summary["Micro-F1"])
    origin = "Macro-F1 ({}) Micro-F1 ({})".format(
        origin_summary["Macro-F1"], origin_summary["Micro-F1"])
    minus_result = "Macro-F1 ({}) Micro-F1 ({})".format(
        summary["Macro-F1"] - origin_summary["Macro-F1"],
        summary["Micro-F1"] - origin_summary["Micro-F1"])
    improve_ratio = "Macro-F1 ({}) Micro-F1 ({})".format(
        (summary["Macro-F1"] - origin_summary["Macro-F1"]) / origin_summary[
            "Macro-F1"],
        (summary["Micro-F1"] - origin_summary["Micro-F1"]) / origin_summary[
            "Micro-F1"])
    return {
        "Original": origin,
        "Improved": improved,
        "Improvement": minus_result,
        "Improvement Ratio": improve_ratio,
    }


def save_summary(summary, explainer):
    """
    Save the summary to a file
    :param summary: a dictionary containing the summary
    :param explainer: an Explainer object
    """
    file_name = f"{explainer.__class__.__name__}_{explainer.model.__class__.__name__}_{explainer.model.dataset.__class__.__name__}_improvement_summary.json"
    folder = "./improvement_summaries/"
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(os.path.join(folder, file_name), "w") as f:
        json.dump(summary, f, indent=4)


if __name__ == "__main__":
    main()
