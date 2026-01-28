import argparse
from utils import str2bool


def getargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='HAN_GCN', help='Model to use')
    parser.add_argument('--dataset_path', type=str, default='../data/ACM',
                        help='Dataset path to use')
    parser.add_argument('--device', type=int, default=0, help='Device to use')
    parser.add_argument('--explainer', type=str, default='GradExplainerMeta',
                        help='Explainer to use')
    parser.add_argument('--times', type=int, default=100, help='Number of times to run')
    parser.add_argument('--save_path', type=str, default='./tmp/ex_at/relation.pkl',
                        help='Path to save the relation')
    parser.add_argument('--ensure_reproducibility', type=str2bool, default=True,
                        help='Ensure reproducibility')
    parser.add_argument('--load_dataset', type=str2bool, default=True,
                        help='Load dataset')
    parser.add_argument('--only_load_test_label_shuffle', type=str2bool, default=True,
                        help='Only load test label shuffle')
    parser.add_argument('--explain_max_nodes', type=int, default=100,
                        help='Max nodes to explain')
    return parser.parse_args()


def set_seed(seed, ensure_reproducibility):
    import torch
    import random
    import numpy as np
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    import os
    # because of the suggestion from
    # https://pytorch.org/docs/stable/notes/randomness.html
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if ensure_reproducibility:
        torch.backends.cudnn.benchmark = False


def train_model(model_name, dataset_name, device, is_load_dataset,
                only_load_test_label_shuffle
                ):
    from datasets import load_dataset
    dataset = load_dataset(dataset_name, is_load_dataset=is_load_dataset,
                           only_load_test_label_shuffle=only_load_test_label_shuffle)
    from models import load_model
    model = load_model(model_name, dataset)
    model.config['summary_path'] = None
    model.to(device)

    print("Training model...")
    model.train()
    model.fit()
    print("Model training finished.")
    return model


def explain(model, explainer_name, device, max_nodes):
    from explainers import load_explainer
    explainer = load_explainer(explainer_name, model.__class__.__name__,
                               model.dataset.__class__.__name__)
    explainer.config['extract_neighbors'] = False
    explainer.to(device)
    print("Explaining model...")
    explainer.explain(model, max_nodes=max_nodes)
    print("Model explanation finished.")
    return explainer


def get_array(explainer, model):
    tmp_path = './tmp/ex_at/model_weight_tmp.pkl'
    import os
    os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
    model.save_attention(tmp_path)
    import torch
    # ensure the model attention is saved
    count = 0
    while True:
        try:
            with open(tmp_path, "rb") as f:
                attention_tmp = torch.load(f)
            del attention_tmp
            break
        except Exception as e:
            count += 1
            if count > 10:
                raise e
            import time
            time.sleep(1)

    attention = torch.load(tmp_path)[0][0]
    attention = torch.softmax(attention, dim=-2)
    attention = attention[:, :].detach().cpu().numpy()

    explanations = explainer.result
    attention = attention[explanations.node_id]

    attention = attention.mean(0)

    import numpy as np
    num_nodes = len(explanations.node_id)
    if num_nodes == 0:
        raise ValueError("No nodes to explain")
    num_meta_paths = len(explanations[0].node_mask)
    # explanation_array1 = np.array(
    #     [[explanations[i].node_mask[0][j] for i in range(100)] for j in
    #      explanations.node_id])
    # explanation_array2 = np.array(
    #     [[explanations[i].node_mask[1][j] for i in range(100)] for j in
    #      explanations.node_id])
    # explanation_array1 = explanation_array1.mean(0)
    # explanation_array2 = explanation_array2.mean(0)
    #
    # return [attention, [explanation_array1, explanation_array2]]
    explanation_array = [
        np.array([[explanations[i].node_mask[k][j] for i in range(num_nodes)] for j in
         explanations.node_id]).mean(0) for k in range(num_meta_paths)]
    explanation_array = np.array(explanation_array)
    return [attention, explanation_array]


def save_result(record, save_path):
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    import pickle
    with open(save_path, 'wb') as f:
        pickle.dump(record, f)


def show_alignment(record):
    from scipy.stats import spearmanr
    from scipy.stats import kendalltau

    attention = [r[0] for r in record]
    explanation = [r[1] for r in record]
    import numpy as np
    attention = np.array(attention)
    explanation = np.array(explanation)
    explanation = np.array([explanation[:, i, :].mean(1) for i in range(explanation.shape[1])])
    num_meta_paths = explanation.shape[0]
    explanation_sum = sum([explanation[i, :] for i in range(num_meta_paths)])
    for i in range(num_meta_paths):
        # print("Kernel tau:",
        print("Meta-path", i, "Kernel tau:",
              kendalltau(attention[:, i], explanation[i, :] / explanation_sum))
        print("Meta-path", i, "Spearman:",
              spearmanr(attention[:, i], explanation[i, :] / explanation_sum))


def main():
    args = getargs()
    record = []
    for time in range(args.times):
        set_seed(time, args.ensure_reproducibility)
        print("Running {}-th time / {} times".format(time + 1, args.times))
        model = train_model(args.model, args.dataset_path, args.device,
                            is_load_dataset=args.load_dataset,
                            only_load_test_label_shuffle=args.only_load_test_label_shuffle
                            )
        explainer = explain(model, args.explainer, args.device,
                            max_nodes=args.explain_max_nodes)
        record.append(get_array(explainer, model))
        del explainer
        import gc
        gc.collect()  # ensure memory of previous explainer is released

        if (time + 1) % 10 == 0:
            print("Show alignment of {}-th time".format(time + 1))
            show_alignment(record)

    save_result(record, args.save_path)


if __name__ == '__main__':
    main()
