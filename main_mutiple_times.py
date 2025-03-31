import argparse
import torch
import numpy as np
import random
import os
import time

from models import load_model
from explainers import load_explainer, load_default_config
from datasets import load_dataset
from utils import str2bool


def getargs():
    parser = argparse.ArgumentParser(description='Explainable AI')
    parser.add_argument('--dataset', type=str, default='../data/ACM',
                        help='Path to dataset, the folder name should be the same as the dataset name')
    parser.add_argument('--model', type=str, default='HAN_GCN',
                        help='Model to use.')
    parser.add_argument('--explainer', type=str, default='GNNExplainerMeta',
                        help='Explainer to use')
    parser.add_argument('--random_seed', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                        help='Random seed. You can try multiple seeds to get overall performance')
    parser.add_argument('--device', type=int, default=0,
                        help='GPU device')
    parser = getargs_optional(parser)
    return parser.parse_args()


def getargs_optional(parser):
    parser.add_argument('--dataset_config', type=str, default=None,
                        help='Dir to dataset config file')
    parser.add_argument('--model_config', type=str, default=None,
                        help='Dir to model config file')
    parser.add_argument('--explainer_config', type=str, default=None,
                        help='Dir to explainer config file')
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
    parser.add_argument('--start_time', type=int, default=None,
                        help='Start time for running')
    parser.add_argument('--explain_max_nodes', type=int, default=100,
                        help='Max number of nodes to explain')
    parser.add_argument('--load_model', type=str2bool, default=False,
                        help='if other explanation model has been run, you can set' + \
                             ' True to load the existing model. ' + \
                             'It will try to load the model from the' + \
                             'same path as the model config')
    parser.add_argument('--load_dataset', type=str2bool, default=True,
                        help='enable to load the existing dataset. If not exist, ' + \
                             'it will try to save the dataset.' + \
                             'It is recommended if you allow shuffling' + \
                             ' in the dataset. However, if it is a large dataset,' + \
                             ' it will double the storage usage. If you want to' + \
                             ' just save the shuffled test label, you can' + \
                             ' set only_load_test_label_shuffle to True')
    parser.add_argument('--only_load_test_label_shuffle', type=str2bool, default=True,
                        help='only load test label shuffle. available only if' + \
                             ' load_dataset is True')
    return parser


def set_seed(seed, ensure_reproducibility=False, times=0):
    if len(seed) > 1:
        seed = seed[times]
    else:
        seed = seed[0]
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


def train_model(model_name, dataset_path, device, dataset_configs=None,
                model_configs=None, is_load_model=False, is_load_dataset=False,
                only_load_test_label_shuffle=False
                ):
    dataset_config = dataset_configs.__next__()
    model_config = model_configs.__next__()
    dataset = load_dataset(dataset_path, dataset_config, is_load_dataset=is_load_dataset,
                           only_load_test_label_shuffle=only_load_test_label_shuffle)
    model = load_model(model_name, dataset, model_config, is_load_model=is_load_model)
    model.to(device)

    if is_load_model:
        print("Use existing model, skip training")
        return model

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
    print("----------------")

    return model


def explain(model, explainer_name, device, explainer_configs=None, minimize=False,
            filter_keys=None,
            max_nodes=None
            ):
    explainer_config = explainer_configs.__next__()
    explainer = load_explainer(explainer_name, model.__class__.__name__,
                               model.dataset.__class__.__name__,
                               explainer_config)
    explainer.to(device)
    result = explainer.explain(model, max_nodes=max_nodes)
    print("Explanation Summary:")
    print("----------------")
    for key, value in result.items():
        print(f"{key}: {value}")
    print("----------------")
    if minimize:
        minimize_keys = ['edge_mask', 'node_id']
        if explainer_name in ['GradExplainerMeta', 'GradExplainerOriginal']:
            minimize_keys.append('node_mask')
        explainer.save_explanation(filter_keys=minimize_keys)
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


def get_times(random_seed, dataset_config, model_config, explainer_config, **kwargs):
    seed_times = 0
    if len(random_seed) > 1:
        seed_times = len(random_seed)

    dataset_times = 0
    if dataset_config is not None:
        if len(os.listdir(dataset_config)) > 1:
            dataset_times = len(os.listdir(dataset_config))

    model_times = 0
    if model_config is not None:
        if len(os.listdir(model_config)) > 1:
            model_times = len(os.listdir(model_config))

    explainer_times = 0
    if explainer_config is not None:
        if len(os.listdir(explainer_config)) > 1:
            explainer_times = len(os.listdir(explainer_config))

    times = 1
    if seed_times != 0:
        times = seed_times

    if dataset_times != 0:
        if dataset_times != times and times != 1:
            raise ValueError("Number of dataset times does not match")
        times = dataset_times

    if model_times != 0:
        if model_times != times and times != 1:
            raise ValueError("Number of model times does not match")
        times = model_times

    if explainer_times != 0:
        if explainer_times != times and times != 1:
            raise ValueError("Number of explainer times does not match")
        times = explainer_times

    # if start_time is None, and seed_times is not 0, then start_time is the first seed
    if kwargs.get("start_time", None) is None:
        if seed_times != 0:
            start_time = random_seed[0]
        else:
            start_time = None
    else:
        start_time = kwargs.get("start_time", None)

    return times, start_time


def multiple_times(name):
    if name is None:
        return [None], False
    if os.listdir(name) == 1:
        return [name], False
    sorted_list = sorted(os.listdir(name))
    return [os.path.join(name, i) for i in sorted_list], True


def main():
    args = getargs()
    times, start_time = get_times(args.random_seed, args.dataset_config,
                                  args.model_config,
                                  args.explainer_config, start_time=args.start_time)
    args.start_time = start_time
    dataset_configs, explainer_configs, model_configs = prepare_configs(
        args.dataset_config,
        args.explainer_config,
        args.model_config,
        times,
        dataset_name=os.path.basename(args.dataset),
        model_name=args.model,
        explainer_name=args.explainer,
        start_time=args.start_time
    )

    if times == 1:
        raise ValueError(
            "This script is for multiple times running, please use main.py for single time running")
    for i in range(times):
        set_seed(args.random_seed, args.ensure_reproducibility, i)
        print("Running {}-th time / {} times".format(i + 1, times))
        model = train_model(args.model, args.dataset, args.device, dataset_configs,
                            model_configs, is_load_model=args.load_model,
                            is_load_dataset=args.load_dataset,
                            only_load_test_label_shuffle=args.only_load_test_label_shuffle
                            )
        if args.save_model:
            model.save()
        if args.save_attention:
            if args.save_gat_attention and model.model_name in ['HAN']:
                model.save_attention(gat_attention=args.save_gat_attention)
            else:
                model.save_attention()
        explainer = explain(model, args.explainer, args.device, explainer_configs,
                            minimize=args.minimize_explanation,
                            filter_keys=args.explanation_keep_keys,
                            max_nodes=args.explain_max_nodes)
        del explainer
        import gc
        gc.collect()  # ensure memory of previous explainer is released
    # visualize(explainer)


def prepare_configs(dataset_config, explainer_config, model_config, times, **kwargs):
    dataset_configs, dataset_is_multiple = multiple_times(dataset_config)
    model_configs, model_is_multiple = multiple_times(model_config)
    explainer_configs, explainer_is_multiple = multiple_times(explainer_config)
    if not dataset_is_multiple:
        dataset_configs = prepare_dataset_configs(dataset_configs, times=times,
                                                  **kwargs)
    else:
        dataset_configs = iter(dataset_configs)
    if not model_is_multiple:
        model_configs = prepare_model_configs(model_configs, times, **kwargs)
    else:
        model_configs = iter(model_configs)
    if not explainer_is_multiple:
        explainer_configs = prepare_explainer_configs(explainer_configs, times,
                                                      **kwargs)
    else:
        explainer_configs = iter(explainer_configs)
    return dataset_configs, explainer_configs, model_configs


def prepare_dataset_configs(dataset_configs, times, **kwargs):
    for t in range(1, times + 1):
        yield dataset_configs[0]


def prepare_model_configs(model_configs, times, **kwargs):
    import json
    path = model_configs[0]
    if path is None:
        path = "model_configs/{model_name}_{dataset_name}.json"
        path = path.format(model_name=kwargs["model_name"],
                           dataset_name=kwargs["dataset_name"])
    with open(path, "r") as f:
        config = json.load(f)

    if kwargs.get("start_time", None) is not None:
        start_time = kwargs["start_time"]
    else:
        start_time = 1

    for t in range(start_time, start_time + times):
        new_config_path = "tmp/model_configs/model_config_{}.json".format(t)
        os.makedirs(os.path.dirname(new_config_path), exist_ok=True)
        new_config = {}
        for key, value in config.items():
            if key.endswith("_path"):
                new_config[key] = os.path.join(os.path.dirname(value),
                                               "{}_{}".format(t,
                                                              os.path.basename(value)))
            else:
                new_config[key] = value
        with open(new_config_path, "w") as f:
            json.dump(new_config, f)
        # ensure the file is written
        count = 0
        while True:
            try:
                with open(new_config_path, "r") as f:
                    config_tmp = json.load(f)
                del config_tmp
                break
            except Exception as e:
                count += 1
                if count > 10:
                    raise e
                time.sleep(1)
        yield new_config_path


def prepare_explainer_configs(explainer_configs, times, **kwargs):
    import json
    path = explainer_configs[0]
    if path is None:
        config = load_default_config(kwargs["explainer_name"],
                                     kwargs["model_name"],
                                     kwargs["dataset_name"])
    else:
        with open(path, "r") as f:
            config = json.load(f)

    if kwargs.get("start_time", None) is not None:
        start_time = kwargs["start_time"]
    else:
        start_time = 1

    for t in range(start_time, start_time + times):
        new_config_path = "tmp/explainer_configs/explainer_config_{}.json".format(t)
        os.makedirs(os.path.dirname(new_config_path), exist_ok=True)
        new_config = {}
        for key, value in config.items():
            if key.endswith("_path"):
                new_config[key] = os.path.join(os.path.dirname(value),
                                               "{}_{}".format(t,
                                                              os.path.basename(value)))
            else:
                new_config[key] = value
        with open(new_config_path, "w") as f:
            json.dump(new_config, f)
        # ensure the file is written
        count = 0
        while True:
            try:
                with open(new_config_path, "r") as f:
                    config_tmp = json.load(f)
                del config_tmp
                break
            except Exception as e:
                count += 1
                if count > 10:
                    raise e
                time.sleep(1)
        yield new_config_path


if __name__ == '__main__':
    main()
