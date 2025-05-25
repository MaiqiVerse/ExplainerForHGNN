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
from utils.retrain_utils import Retrain


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
    parser.add_argument('--topk', type=float, default=0.75,
                        help='Top k nodes to be explained')
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
    model = load_model(model_name, dataset, model_config, device=device)
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


def explain(model, explainer_name, device, explainer_config=None, topk=0.75
            ):
    explainer = load_explainer(explainer_name, model.__class__.__name__,
                               model.dataset.__class__.__name__,
                               explainer_config, device=device)
    if "top_k_for_edge_mask" in explainer.config:
        explainer.config["top_k_for_edge_mask"] = topk
    if "top_k_for_feature_mask" in explainer.config:
        explainer.config["top_k_for_feature_mask"] = topk
    if not explainer.config.get("top_k_for_edge_mask", False) and not explainer.config.get(
            "top_k_for_feature_mask", False):
        print("Please check the setting is correct, top_k_for_edge_mask and top_k_for_feature_mask are both not exist")
    explainer.to(device)
    # Here we consider the explanation for all nodes, not just the test nodes
    result_dict, result_nodes = explain_model(explainer, model)
    retrainer = Retrain(explainer, result_nodes, result_dict)
    _, retrain_result = retrainer.fit()
    return explainer, retrain_result


def explain_model(explainer, model):
    """
    Explain the model using the explainer
    :param explainer: an Explainer object
    :param model: a Model object
    :return: the result of the explanation
    """
    explainer.model = model
    result = []
    result_dict = {}
    result_nodes = {}
    train_labels = explainer.model.dataset.labels[0]
    val_labels = explainer.model.dataset.labels[1]
    test_labels = explainer.model.dataset.labels[2]
    for idx, label in train_labels + val_labels + test_labels:
        explain_node = explainer.__class__(explainer.config)
        explain_node.to(explainer.device)
        explanation = explain_node.explain(model, node_id=idx)
        result.append(explanation)
        result_nodes[idx] = explain_node
        result_dict[idx] = explanation
    result = explainer.construct_explanation(result)
    explainer.result = result
    explainer.evaluate()
    explainer.save_summary()
    return result_dict, result_nodes



def main():
    args = get_args()
    set_seed(args.random_seed, True)

    # Set device
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # Load model
    model = train_model(args.model, args.dataset, device)

    # Get the explanation for the target node
    explainer_config = prepare_improve_acc_explainer_config(args.explainer, args.model, args.dataset)
    explainer, retrain_result = explain(model, args.explainer, args.device, explainer_config, args.topk)

    # Get the improvement summary
    improvement_summary = get_improvement_summary(explainer, retrain_result)

    # Print the improvement summary
    print("Improvement Summary:")
    print("----------------")
    for key, value in improvement_summary.items():
        print(f"{key}: {value}")

    # Save the summary
    save_summary(improvement_summary, explainer, args.topk)


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


def get_improvement_summary(explainer, retrain_result):
    """
    Get the improvement summary from the explainer
    :param explainer: an Explainer object
    :return: a dictionary containing the improvement summary
    """
    summary = explainer.get_summary()
    origin_summary = explainer.model.get_summary()
    improved = "Macro-F1 ({:.4f}) Micro-F1 ({:.4f})".format(
        summary["Macro-F1"], summary["Micro-F1"])
    retain_improved = "Macro-F1 ({:.4f}) Micro-F1 ({:.4f})".format(
        retrain_result["Macro-F1"], retrain_result["Micro-F1"])
    origin = "Macro-F1 ({:.4f}) Micro-F1 ({:.4f})".format(
        origin_summary["Macro-F1"], origin_summary["Micro-F1"])
    minus_result = "Macro-F1 ({:.4f}) Micro-F1 ({:.4f})".format(
        summary["Macro-F1"] - origin_summary["Macro-F1"],
        summary["Micro-F1"] - origin_summary["Micro-F1"])
    minus_result_retain = "Macro-F1 ({:.4f}) Micro-F1 ({:.4f})".format(
        retrain_result["Macro-F1"] - origin_summary["Macro-F1"],
        retrain_result["Micro-F1"] - origin_summary["Micro-F1"])
    improve_ratio = "Macro-F1 ({:.4f}) Micro-F1 ({:.4f})".format(
        (summary["Macro-F1"] - origin_summary["Macro-F1"]) / origin_summary[
            "Macro-F1"],
        (summary["Micro-F1"] - origin_summary["Micro-F1"]) / origin_summary[
            "Micro-F1"])
    improve_ratio_retain = "Macro-F1 ({:.4f}) Micro-F1 ({:.4f})".format(
        (retrain_result["Macro-F1"] - origin_summary["Macro-F1"]) / origin_summary[
            "Macro-F1"],
        (retrain_result["Micro-F1"] - origin_summary["Micro-F1"]) / origin_summary[
            "Micro-F1"])
    return {
        "Original": origin,
        "Improved": improved,
        "Retain Improved": retain_improved,
        "Improvement": minus_result,
        "Retain Improvement": minus_result_retain,
        "Improvement Ratio": improve_ratio,
        "Retain Improvement Ratio": improve_ratio_retain,
    }


def save_summary(summary, explainer, topk=0.75):
    """
    Save the summary to a file
    :param summary: a dictionary containing the summary
    :param explainer: an Explainer object
    """
    file_name = f"{explainer.__class__.__name__}_{explainer.model.__class__.__name__}_{explainer.model.dataset.__class__.__name__}_topk_{topk}_improvement_summary.json"
    folder = "./improvement_summaries_retrain/"
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(os.path.join(folder, file_name), "w") as f:
        json.dump(summary, f, indent=4)


if __name__ == "__main__":
    main()
