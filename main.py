import argparse
import torch
import numpy as np
import random

from models import load_model
from explainers import load_explainer
from datasets import load_dataset


def getargs():
    parser = argparse.ArgumentParser(description='Explainable AI')
    parser.add_argument('--dataset', type=str, default='./data/ACM',
                        help='Path to dataset, the folder name should be the same as the dataset name')
    parser.add_argument('--model', type=str, default='HAN',
                        help='Model to use. \n./model_configs/{Model}_{Dataset}.json will be used for configuration')
    parser.add_argument('--explainer', type=str, default='GNNExplainerMeta',
                        help='Explainer to use\n./explainer_configs/{Explainer}_{Model}_{Dataset}.json will be used for configuration')
    parser.add_argument('--random_seed', type=int, default=0,
                        help='Random seed. You can try multiple seeds to get overall performance')
    parser.add_argument('--device', type=int, default=0,
                        help='GPU device')
    parser = getargs_optional(parser)
    return parser.parse_args()


def getargs_optional(parser):
    parser.add_argument('--dataset_config', type=str, default=None,
                        help='Path to dataset config file')
    parser.add_argument('--model_config', type=str, default=None,
                        help='Path to model config file')
    parser.add_argument('--explainer_config', type=str, default=None,
                        help='Path to explainer config file')
    parser.add_argument('--ensure_reproducibility', action='store_true',
                        default=False,
                        help='Ensure reproducibility')
    parser.add_argument('--save_attention', action='store_true',
                        default=False,
                        help='Save attention weights')
    parser.add_argument('--save_model', action='store_true',
                        default=False,
                        help='Save model')
    parser.add_argument('--minimize_explanation', action='store_true',
                        default=False,
                        help='Only save edge mask')
    parser.add_argument('--save_gat_attention', action='store_true',
                        default=False,
                        help='Save GAT attention weights')
    parser.add_argument('--explanation_keep_keys', type=str, nargs='+', default=[],
                        help='Keys to keep in explanation')
    parser.add_argument('--explain_max_nodes', type=int, default=None,
                        help='Maximum number of nodes to explain')
    return parser


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


def explain(model, explainer_name, device, explainer_config=None, minimize=False,
            filter_keys=None,
            max_nodes=None
            ):
    explainer = load_explainer(explainer_name, model.__class__.__name__,
                               model.dataset.__class__.__name__,
                               explainer_config,
                               device=device)
    explainer.to(device)
    result = explainer.explain(model, max_nodes=max_nodes)
    print("Explanation Summary:")
    print("----------------")
    for key, value in result.items():
        print(f"{key}: {value}")
    if minimize:
        explainer.save_explanation(filter_keys=['edge_mask'])
    elif filter_keys is not None and len(filter_keys) > 0:
        explainer.save_explanation(filter_keys=filter_keys)
    else:
        explainer.save_explanation()
    return explainer


# Do not implement now
# def visualize(explainer):
#     if explainer.metrics is not None:
#         explainer.get_metrics()
#         explainer.save_metrics()
#
#     if explainer.model.summary is not None:
#         explainer.model.get_summary()
#         explainer.model.save_summary()
#
#     explainer.visualize()


def main():
    args = getargs()
    set_seed(args.random_seed, args.ensure_reproducibility)
    model = train_model(args.model, args.dataset, args.device, args.dataset_config,
                        args.model_config)
    if args.save_model:
        model.save()
    if args.save_attention:
        if args.save_gat_attention and model.model_name in ['HAN']:
            model.save_attention(gat_attention=args.save_gat_attention)
        else:
            model.save_attention()
    explainer = explain(model, args.explainer, args.device, args.explainer_config,
                        minimize=args.minimize_explanation,
                        filter_keys=args.explanation_keep_keys,
                        max_nodes=args.explain_max_nodes)
    # visualize(explainer)


if __name__ == '__main__':
    main()
