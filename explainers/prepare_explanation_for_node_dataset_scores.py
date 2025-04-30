import torch


def get_edge_mask_hard(explainer, opposite=False):
    if getattr(explainer, 'edge_mask_for_output', None) is None:
        gs = explainer.extract_neighbors_input()[0]
        edge_mask = []
        for g in gs:
            edge_mask.append(torch.ones(g._nnz(), device=explainer.model.device))
        if explainer.config['edge_mask_hard_method'] in ['threshold', 'auto_threshold']:
            raise ValueError('edge_mask_hard_method is threshold or auto_threshold, '
                             'but edge_mask_for_output is not found')
    else:
        edge_mask = explainer.edge_mask_for_output
    if explainer.config['edge_mask_hard_method'] == 'threshold':
        # edge_mask = explainer.edge_mask_for_output
        threshold = explainer.config['edge_mask_threshold']
        if opposite:
            return [em <= threshold for em in edge_mask]
        return [em > threshold for em in edge_mask]
    elif explainer.config['edge_mask_hard_method'] == 'auto_threshold':
        # edge_mask = explainer.edge_mask_for_output
        edge_mask_concat = torch.cat(edge_mask)
        threshold = torch.quantile(edge_mask_concat,
                                   explainer.config['threshold_percentage_edge'])
        if opposite:
            return [em <= threshold for em in edge_mask]
        return [em > threshold for em in edge_mask]
    elif explainer.config['edge_mask_hard_method'] == 'original':
        if opposite:
            return [1 - em.float() for em in edge_mask]
        return edge_mask
    elif explainer.config['edge_mask_hard_method'] == 'top_k':
        top_k = explainer.config['top_k_for_edge_mask']
        return get_top_k_edge_mask_core(edge_mask, top_k, opposite)
    else:
        raise ValueError('Invalid edge_mask_threshold_method: {}'.format(
            explainer.config['edge_mask_threshold_method']))


def get_top_k_edge_mask_core(edge_mask, top_k, opposite=False, separate=True):
    # Compute sizes and cumulative sizes of edge masks
    size_edge_mask = [em.numel() for em in edge_mask]

    if not separate:
        # Concatenate and sort to get the top_k indices
        edge_mask_concat = torch.cat(edge_mask)
        top_k = int(top_k * edge_mask_concat.size(0))
        _, indices = torch.topk(edge_mask_concat, top_k)

        # Initialize the edge mask hard tensor as a single concatenated tensor
        edge_mask_hard_concat = torch.zeros_like(
            edge_mask_concat) if not opposite else torch.ones_like(edge_mask_concat)

        # Set the top_k positions to 1 (or 0 if opposite=True)
        edge_mask_hard_concat[indices] = 1 if not opposite else 0

        # Split the hard mask back into the original list of tensors
        edge_mask_hard = torch.split(edge_mask_hard_concat, size_edge_mask)

        return list(edge_mask_hard)

    edge_mask_hard = []
    top_ori = top_k
    for em in edge_mask:
        # Sort the edge mask tensor to get the top_k indices
        top_k = int(top_ori * em.size(0))
        _, indices = torch.topk(em, top_k)

        # Initialize the edge mask hard tensor
        em_hard = torch.zeros_like(em) if not opposite else torch.ones_like(em)

        # Set the top_k positions to 1 (or 0 if opposite=True)
        em_hard[indices] = 1 if not opposite else 0

        edge_mask_hard.append(em_hard)

    return list(edge_mask_hard)


def get_masked_gs_hard(explainer, opposite=False):
    edge_mask_hard = get_edge_mask_hard(explainer, opposite)
    gs = explainer.neighbor_input['gs']
    masked_gs_hard = []
    for g, em in zip(gs, edge_mask_hard):
        g = g.coalesce()
        mask = torch.sparse_coo_tensor(g.indices(), em,
                                       g.size())
        masked_g = g * mask
        masked_g = masked_g.coalesce()
        masked_gs_hard.append(masked_g)

    return masked_gs_hard


def get_masked_gs_hard_core(gs, edge_mask_hard):
    masked_gs_hard = []
    for g, em in zip(gs, edge_mask_hard):
        g = g.coalesce()
        values = g.values()
        mask = em
        masked_g = torch.sparse_coo_tensor(g.indices()[:, mask], values[mask],
                                           g.size())
        masked_gs_hard.append(masked_g)

    return masked_gs_hard


def full_feature_mask_hard(feature_mask, explainer, opposite=False, separate=True):
    if explainer.config['feature_mask_hard_method'] == 'threshold':
        threshold = explainer.config['feature_mask_threshold']
        if isinstance(feature_mask, list):
            if opposite:
                return [fm <= threshold for fm in feature_mask]
            return [fm > threshold for fm in feature_mask]
        else:
            if opposite:
                return feature_mask <= threshold
            return feature_mask > threshold
    elif explainer.config['feature_mask_hard_method'] == 'auto_threshold':
        if isinstance(feature_mask, list):
            if separate:
                result = []
                for fm in feature_mask:
                    threshold = torch.quantile(fm, explainer.config['threshold_percentage_feature'])
                    if opposite:
                        result.append(fm <= threshold)
                    else:
                        result.append(fm > threshold)
                return result
            else:
                feature_mask_tensor = torch.cat(feature_mask)
                threshold = torch.quantile(feature_mask_tensor,
                                           explainer.config['threshold_percentage_feature'])
                if opposite:
                    return [fm <= threshold for fm in feature_mask]
                return [fm > threshold for fm in feature_mask]
        else:
            threshold = torch.quantile(feature_mask,
                                       explainer.config['threshold_percentage_feature'])
            if opposite:
                return feature_mask <= threshold
            return feature_mask > threshold
    elif explainer.config['feature_mask_hard_method'] == 'original':
        if isinstance(feature_mask, list):
            if opposite:
                return [1 - fm.float() for fm in feature_mask]
            return [fm.float() for fm in feature_mask]
        else:
            if opposite:
                return 1 - feature_mask.float()
            return feature_mask.float()

    elif explainer.config['feature_mask_hard_method'] == 'top_k':
        top_k = explainer.config['top_k_for_feature_mask']
        if isinstance(feature_mask, list):
            if separate:
                return [get_top_k_feature_mask_core(fm, top_k, opposite) for fm in
                        feature_mask]
            else:
                feature_mask_tensor = torch.cat(feature_mask)
                feature_mask_hard = get_top_k_feature_mask_core(feature_mask_tensor,
                                                                top_k,
                                                                opposite)
                return list(
                    torch.split(feature_mask_hard, [len(fm) for fm in feature_mask]))
        top_k = int(top_k * torch.numel(feature_mask))
        # if the shape of feature_mask is 2D, we need to flatten it first
        flatten = False
        ori_shape = None
        if len(feature_mask.shape) > 1:
            ori_shape = feature_mask.shape
            feature_mask = feature_mask.flatten()
            flatten = True
        indices = torch.sort(feature_mask, descending=True)[1][:top_k]

        if opposite:
            feature_mask_hard = torch.ones_like(feature_mask)
            feature_mask_hard[indices] = 0
        else:
            feature_mask_hard = torch.zeros_like(feature_mask)
            feature_mask_hard[indices] = 1
        if flatten:
            feature_mask_hard = feature_mask_hard.reshape(ori_shape)
        return feature_mask_hard

    else:
        raise ValueError('Invalid feature_mask_threshold_method: {}'.format(
            explainer.config['feature_mask_threshold_method']))


def get_feature_mask_hard(explainer, opposite=False, separate=True):
    if explainer.feature_mask_for_output is None:
        return None

    feature_mask = explainer.feature_mask_for_output
    if not isinstance(feature_mask, list) and len(feature_mask.shape) == 2:
        return full_feature_mask_hard(feature_mask, explainer, opposite, separate)

    if explainer.config['feature_mask_hard_method'] == 'threshold':
        feature_mask = explainer.feature_mask_for_output
        threshold = explainer.config['feature_mask_threshold']
        if isinstance(feature_mask, list):
            if opposite:
                return [fm <= threshold for fm in feature_mask]
            return [fm > threshold for fm in feature_mask]
        if opposite:
            return feature_mask <= threshold
        return feature_mask > threshold
    elif explainer.config['feature_mask_hard_method'] == 'auto_threshold':
        feature_mask = explainer.feature_mask_for_output
        if isinstance(feature_mask, list):
            if separate:
                result = []
                for fm in feature_mask:
                    threshold = torch.quantile(fm, explainer.config['threshold_percentage_feature'])
                    if opposite:
                        result.append(fm <= threshold)
                    else:
                        result.append(fm > threshold)
                return result
            else:
                feature_mask_tensor = torch.cat(feature_mask)
                threshold = torch.quantile(feature_mask_tensor,
                                           explainer.config['threshold_percentage_feature'])
                if opposite:
                    return [
                        fm <= threshold for fm in feature_mask
                    ]
                return [
                    fm > threshold for fm in feature_mask
                ]
        else:
            threshold = torch.quantile(feature_mask,
                                       explainer.config['threshold_percentage_feature'])
            if opposite:
                return feature_mask <= threshold
            return feature_mask > threshold
    elif explainer.config['feature_mask_hard_method'] == 'original':
        if isinstance(feature_mask, list):
            if opposite:
                return [1 - fm.float() for fm in feature_mask]
            return [fm.float() for fm in feature_mask]
        else:
            if opposite:
                return 1 - explainer.feature_mask_for_output.float()
            return explainer.feature_mask_for_output.float()
    elif explainer.config['feature_mask_hard_method'] == 'top_k':
        feature_mask = explainer.feature_mask_for_output
        top_k = explainer.config['top_k_for_feature_mask']
        if isinstance(feature_mask, list):
            if separate:
                return [get_top_k_feature_mask_core(fm, top_k, opposite) for fm in
                        feature_mask]
            else:
                feature_mask_tensor = torch.cat(feature_mask)
                feature_mask_hard = get_top_k_feature_mask_core(feature_mask_tensor,
                                                                top_k,
                                                                opposite)
                return list(
                    torch.split(feature_mask_hard, [len(fm) for fm in feature_mask]))
        top_k = int(top_k * len(feature_mask))
        indices = torch.sort(feature_mask, descending=True)[1][:top_k]
        if opposite:
            feature_mask_hard = torch.ones_like(feature_mask)
            feature_mask_hard[indices] = 0
        else:
            feature_mask_hard = torch.zeros_like(feature_mask)
            feature_mask_hard[indices] = 1
        return feature_mask_hard
    else:
        raise ValueError('Invalid feature_mask_threshold_method: {}'.format(
            explainer.config['feature_mask_threshold_method']))


def fidelity_neg_explanation(node_explanation, explainer):
    """
    Set the masked_pred_label_hard, pred_label_hard and label for the fidelity- score
    :param node_explanation: a NodeExplanation object
    :param explainer: an Explainer object
    """
    if "masked_pred_label_hard" not in node_explanation:
        if "masked_pred_label" not in node_explanation:
            flag = True
            if "masked_gs_hard" not in node_explanation and getattr(explainer,
                                                                    'edge_mask_for_output',
                                                                    None) is not None:
                masked_gs_hard = get_masked_gs_hard(explainer)
                node_explanation.masked_gs_hard = masked_gs_hard
                flag = False
            elif "masked_gs_hard" in node_explanation:
                flag = False
            else:
                node_explanation.masked_gs_hard = None

            if "feature_mask_hard" not in node_explanation and getattr(explainer,
                                                                       'feature_mask_for_output',
                                                                       None) is not None:
                feature_mask_hard = get_feature_mask_hard(explainer)
                node_explanation.feature_mask_hard = feature_mask_hard
                flag = False
            elif "feature_mask_hard" in node_explanation:
                flag = False
            else:
                node_explanation.feature_mask_hard = None
            if flag:
                raise ValueError('masked_gs_hard and feature_mask_hard are not found')

            masked_pred_label = \
                explainer.model.custom_forward(explainer.get_custom_input_handle_fn(
                    node_explanation.masked_gs_hard,
                    node_explanation.feature_mask_hard))[
                    explainer.mapping_node_id()]
            node_explanation.masked_pred_label = masked_pred_label

        masked_pred_label_hard = node_explanation.masked_pred_label.argmax()
        node_explanation.masked_pred_label_hard = masked_pred_label_hard

    if "pred_label_hard" not in node_explanation:
        if "pred_label" not in node_explanation:
            pred_label = explainer.model()[explainer.node_id]
            node_explanation.pred_label = pred_label

        pred_label_hard = node_explanation.pred_label.argmax()
        node_explanation.pred_label_hard = pred_label_hard

    if "label" not in node_explanation:
        label_test = explainer.model.dataset.labels[2]
        label = None
        for l in label_test:
            if l[0] == explainer.node_id:
                label = l[1]
                break

        label = torch.tensor(label, device=explainer.model.device)
        node_explanation.label = label

    return node_explanation


def fidelity_pos_explanation(node_explanation, explainer):
    """
    Set the opposite_masked_pred_label_hard, opposite_masked_pred_label and label for the fidelity+ score
    :param node_explanation: a NodeExplanation object
    :param explainer: an Explainer object
    """
    if "opposite_masked_pred_label_hard" not in node_explanation:
        if "opposite_masked_pred_label" not in node_explanation:
            flag = True
            if "opposite_masked_gs_hard" not in node_explanation and getattr(
                    explainer, 'edge_mask_for_output', None) is not None:
                opposite_masked_gs_hard = get_masked_gs_hard(explainer, opposite=True)
                node_explanation.opposite_masked_gs_hard = opposite_masked_gs_hard
                flag = False
            elif "opposite_masked_gs_hard" in node_explanation:
                flag = False
            else:
                node_explanation.opposite_masked_gs_hard = None

            if "opposite_feature_mask_hard" not in node_explanation and getattr(
                    explainer, 'feature_mask_for_output', None) is not None:
                opposite_feature_mask_hard = get_feature_mask_hard(explainer,
                                                                   opposite=True)
                node_explanation.opposite_feature_mask_hard = opposite_feature_mask_hard
                flag = False
            elif "opposite_feature_mask_hard" in node_explanation:
                flag = False
            else:
                node_explanation.opposite_feature_mask_hard = None
            if flag:
                raise ValueError(
                    'opposite_masked_gs_hard and opposite_feature_mask_hard are not found')

            opposite_masked_pred_label = \
                explainer.model.custom_forward(explainer.get_custom_input_handle_fn(
                    node_explanation.opposite_masked_gs_hard,
                    node_explanation.opposite_feature_mask_hard))[
                    explainer.mapping_node_id()]
            node_explanation.opposite_masked_pred_label = opposite_masked_pred_label

        opposite_masked_pred_label_hard = node_explanation.opposite_masked_pred_label.argmax()
        node_explanation.opposite_masked_pred_label_hard = opposite_masked_pred_label_hard

    if "pred_label_hard" not in node_explanation:
        if "pred_label" not in node_explanation:
            pred_label = explainer.model()[explainer.node_id]
            node_explanation.pred_label = pred_label

        pred_label_hard = node_explanation.pred_label.argmax()
        node_explanation.pred_label_hard = pred_label_hard

    if "label" not in node_explanation:
        label_test = explainer.model.dataset.labels[2]
        label = None
        for l in label_test:
            if l[0] == explainer.node_id:
                label = l[1]
                break

        label = torch.tensor(label, device=explainer.model.device)
        node_explanation.label = label

    return node_explanation


def characterization_score_explanation(node_explanation, explainer):
    """
    Set the necessary data for the characterization score
    :param node_explanation: a NodeExplanation object
    :param explainer: an Explainer object
    """

    node_explanation = fidelity_pos_explanation(node_explanation, explainer)
    node_explanation = fidelity_neg_explanation(node_explanation, explainer)

    return node_explanation


def get_multi_edge_threshold_auc(explainer):
    if explainer.config['edge_mask_hard_method'] == 'threshold':
        if explainer.config.get(['threshold_auc'], None) is not None:
            return explainer.config['threshold_auc']
        else:
            num = 11
            return [i / (num - 1) for i in range(num)]
    elif explainer.config['edge_mask_hard_method'] == 'auto_threshold':
        edge_mask = explainer.edge_mask_for_output
        edge_mask_concat = torch.cat(edge_mask)
        threshold = [torch.quantile(edge_mask_concat, i) for i in
                     explainer.config['threshold_percentage_auc']]
        return threshold
    elif explainer.config['edge_mask_hard_method'] == 'original':
        if explainer.config.get(['threshold_auc'], None) is not None:
            return explainer.config['threshold_auc']
        else:
            num = 11
            return [i / (num - 1) for i in range(num)]
    elif explainer.config['edge_mask_hard_method'] == 'top_k':
        if explainer.config.get(['top_k_for_auc'], None) is not None:
            return explainer.config['top_k_for_auc']
        else:
            num = 11
            return [i / (num - 1) for i in range(num)]
    else:
        raise ValueError('Invalid edge_mask_threshold_method: {}'.format(
            explainer.config['edge_mask_threshold_method']))


def get_edge_mask_hard_threshold_auc(explainer, opposite=False):
    threshold_values = get_multi_edge_threshold_auc(explainer)
    edge_mask = explainer.edge_mask_for_output
    if explainer.config['edge_mask_hard_method'] == 'top_k':
        edge_masks_hard = []
        for threshold in threshold_values:
            edge_mask_hard = get_top_k_edge_mask_core(edge_mask, threshold, opposite)
            edge_masks_hard.append(edge_mask_hard)
        return edge_masks_hard
    else:
        if not opposite:
            return [[em > threshold for em in edge_mask] for threshold in
                    threshold_values]
        else:
            return [[em <= threshold for em in edge_mask] for threshold in
                    threshold_values]


def get_masked_gs_hard_threshold_auc(explainer, opposite=False):
    gs = explainer.neighbor_input['gs']
    masked_gs_threshold = []
    edge_masks = get_edge_mask_hard_threshold_auc(explainer, opposite)
    for edge_mask_hard in edge_masks:
        masked_gs_hard = get_masked_gs_hard_core(gs, edge_mask_hard)
        masked_gs_threshold.append(masked_gs_hard)
    return masked_gs_threshold


def get_top_k_feature_mask_core(feature_mask, top_k, opposite=False):
    top_k = int(top_k * len(feature_mask))
    indices = torch.sort(feature_mask, descending=True)[1][:top_k]
    if opposite:
        feature_mask_hard = torch.ones_like(feature_mask)
        feature_mask_hard[indices] = 0
    else:
        feature_mask_hard = torch.zeros_like(feature_mask)
        feature_mask_hard[indices] = 1
    return feature_mask_hard


def get_multi_feature_threshold_auc(explainer):
    if explainer.config['feature_mask_hard_method'] == 'threshold':
        if explainer.config.get(['threshold_auc'], None) is not None:
            return explainer.config['threshold_auc']
        else:
            num = 11
            return [i / (num - 1) for i in range(num)]
    elif explainer.config['feature_mask_hard_method'] == 'auto_threshold':
        feature_mask = explainer.feature_mask_for_output
        threshold = [torch.quantile(feature_mask, i) for i in
                     explainer.config['threshold_percentage_auc']]
        return threshold
    elif explainer.config['feature_mask_hard_method'] == 'original':
        if explainer.config.get(['threshold_auc'], None) is not None:
            return explainer.config['threshold_auc']
        else:
            num = 11
            return [i / (num - 1) for i in range(num)]
    elif explainer.config['feature_mask_hard_method'] == 'top_k':
        if explainer.config.get(['top_k_for_auc'], None) is not None:
            return explainer.config['top_k_for_auc']
        else:
            num = 11
            return [i / (num - 1) for i in range(num)]
    else:
        raise ValueError('Invalid feature_mask_threshold_method: {}'.format(
            explainer.config['feature_mask_threshold_method']))


def get_feature_mask_hard_threshold_auc(explainer, opposite=False):
    threshold_values = get_multi_feature_threshold_auc(explainer)
    feature_mask = explainer.feature_mask_for_output
    if explainer.config['feature_mask_hard_method'] == 'top_k':
        feature_masks_hard = []
        for threshold in threshold_values:
            feature_mask_hard = get_top_k_feature_mask_core(feature_mask, threshold,
                                                            opposite)
            feature_masks_hard.append(feature_mask_hard)
        return feature_masks_hard
    else:
        if not opposite:
            return [feature_mask > threshold for threshold in threshold_values]
        else:
            return [feature_mask <= threshold for threshold in threshold_values]


def fidelity_curve_auc_explanation(node_explanation, explainer):
    """

    :param node_explanation: a NodeExplanation object
    :param explainer: an Explainer object
    """
    if "label" not in node_explanation:
        label_test = explainer.model.dataset.labels[2]
        label = None
        for l in label_test:
            if l[0] == explainer.node_id:
                label = l[1]
                break

        label = torch.tensor(label, device=explainer.model.device)
        node_explanation.label = label

    if "pred_label_hard" not in node_explanation:
        if "pred_label" not in node_explanation:
            pred_label = explainer.model()[explainer.node_id]
            node_explanation.pred_label = pred_label

        pred_label_hard = node_explanation.pred_label.argmax()
        node_explanation.pred_label_hard = pred_label_hard

    if "masked_pred_labels_hard_threshold" not in node_explanation:
        if not explainer.config.get('auc_use_feature_mask', False) and \
                not explainer.config.get('auc_use_edge_mask', False):
            raise ValueError(
                'one of auc_use_feature_mask and auc_use_edge_mask should be True')
        if explainer.config.get('auc_use_feature_mask', False):
            feature_masks_hard = get_feature_mask_hard_threshold_auc(explainer)
        else:
            feature_masks_hard = None
        if explainer.config.get('auc_use_edge_mask', False):
            masked_gs = get_masked_gs_hard_threshold_auc(explainer)
        else:
            masked_gs = None
        masked_pred_labels_hard_threshold = []
        length = len(feature_masks_hard) if feature_masks_hard is not None else len(
            masked_gs)
        for i in range(length):
            g = masked_gs[i] if masked_gs is not None else None
            feature_mask = feature_masks_hard[
                i] if feature_masks_hard is not None else None
            masked_pred_label = \
                explainer.model.custom_forward(explainer.get_custom_input_handle_fn(
                    g, feature_mask))[
                    explainer.mapping_node_id()]
            masked_pred_label_hard = masked_pred_label.argmax()
            masked_pred_labels_hard_threshold.append(masked_pred_label_hard)

        node_explanation.masked_pred_labels_hard_threshold = masked_pred_labels_hard_threshold

    if "opposite_masked_pred_labels_hard_threshold" not in node_explanation:
        if not explainer.config.get('auc_use_feature_mask', False) and \
                not explainer.config.get('auc_use_edge_mask', False):
            raise ValueError(
                'one of auc_use_feature_mask and auc_use_edge_mask should be True')
        if explainer.config.get('auc_use_feature_mask', False):
            feature_masks_hard = get_feature_mask_hard_threshold_auc(explainer,
                                                                     opposite=True)
        else:
            feature_masks_hard = None
        if explainer.config.get('auc_use_edge_mask', False):
            masked_gs = get_masked_gs_hard_threshold_auc(explainer, opposite=True)
        else:
            masked_gs = None
        masked_pred_labels_hard_threshold = []
        length = len(feature_masks_hard) if feature_masks_hard is not None else len(
            masked_gs)
        for i in range(length):
            g = masked_gs[i] if masked_gs is not None else None
            feature_mask = feature_masks_hard[
                i] if feature_masks_hard is not None else None
            masked_pred_label = \
                explainer.model.custom_forward(explainer.get_custom_input_handle_fn(
                    g, feature_mask))[
                    explainer.mapping_node_id()]
            masked_pred_label_hard = masked_pred_label.argmax()
            masked_pred_labels_hard_threshold.append(masked_pred_label_hard)

        node_explanation.opposite_masked_pred_labels_hard_threshold = masked_pred_labels_hard_threshold

    return node_explanation


def unfaithfulness_explanation(node_explanation, explainer):
    """
    Set the masked_pred_label and pred_label for the unfaithfulness score
    :param node_explanation: a NodeExplanation object
    :param explainer: an Explainer object
    """
    if "masked_pred_label" not in node_explanation:
        flag = True
        if "masked_gs_hard" not in node_explanation and getattr(explainer,
                                                                'edge_mask_for_output',
                                                                None) is not None:
            masked_gs_hard = get_masked_gs_hard(explainer)
            node_explanation.masked_gs_hard = masked_gs_hard
            flag = False
        elif "masked_gs_hard" in node_explanation:
            flag = False
        else:
            node_explanation.masked_gs_hard = None
        if "feature_mask_hard" not in node_explanation and getattr(explainer,
                                                                   'feature_mask_for_output',
                                                                   None) is not None:
            feature_mask_hard = get_feature_mask_hard(explainer)
            node_explanation.feature_mask_hard = feature_mask_hard
            flag = False
        elif "feature_mask_hard" in node_explanation:
            flag = False
        else:
            node_explanation.feature_mask_hard = None
        if flag:
            raise ValueError('masked_gs_hard and feature_mask_hard are not found')
        masked_pred_label = \
            explainer.model.custom_forward(explainer.get_custom_input_handle_fn(
                node_explanation.masked_gs_hard,
                node_explanation.feature_mask_hard))[
                explainer.mapping_node_id()]
        node_explanation.masked_pred_label = masked_pred_label

    if "pred_label" not in node_explanation:
        pred_label = explainer.model()[explainer.node_id]
        node_explanation.pred_label = pred_label

    return node_explanation


def sparsity_explanation(node_explanation, explainer):
    """
    Set the edge_mask_hard for the sparsity score
    :param node_explanation: a NodeExplanation object
    :param explainer: an Explainer object
    """
    if explainer.config.get('sparsity_type', 'edge') == 'edge' and "edge_mask_hard" not in node_explanation and getattr(
            explainer,
            'edge_mask_for_output',
            None) is not None:
        edge_mask_hard = get_edge_mask_hard(explainer)
        node_explanation.edge_mask_hard = edge_mask_hard

    if explainer.config.get('sparsity_type', 'edge') != 'edge' and "feature_mask_hard" not in node_explanation and getattr(
            explainer,
            'feature_mask_for_output',
            None) is not None:
        feature_mask_hard = get_feature_mask_hard(explainer)
        node_explanation.feature_mask_hard = feature_mask_hard

    return node_explanation


def generate_perturbed_feature_mask(feature_mask, k):
    top_k = int(k * len(feature_mask))
    top_k_indices = torch.argsort(feature_mask, descending=True)[top_k:]
    perturbed_feature_mask = feature_mask.clone()
    perturbed_feature_mask[top_k_indices] = torch.normal(0, 0.1,
                                                         size=(len(top_k_indices),),
                                                         device=feature_mask.device)
    return perturbed_feature_mask


def graph_exp_faith_feature_explanation(node_explanation, explainer):
    """
    Set the masked_pred_label and perturbed_feature_pred_label for the faithfulness score

    :param node_explanation: a NodeExplanation object
    :param explainer: an Explainer object
    """
    if "masked_pred_label" not in node_explanation:
        flag = True
        if "masked_gs_hard" not in node_explanation and getattr(explainer,
                                                                'edge_mask_for_output',
                                                                None) is not None:
            masked_gs_hard = get_masked_gs_hard(explainer)
            node_explanation.masked_gs_hard = masked_gs_hard
            flag = False
        elif "masked_gs_hard" in node_explanation:
            flag = False
        else:
            node_explanation.masked_gs_hard = None
        if "feature_mask_hard" not in node_explanation and getattr(explainer,
                                                                   'feature_mask_for_output',
                                                                   None) is not None:
            feature_mask_hard = get_feature_mask_hard(explainer)
            node_explanation.feature_mask_hard = feature_mask_hard
            flag = False
        elif "feature_mask_hard" in node_explanation:
            flag = False
        else:
            node_explanation.feature_mask_hard = None
        if flag:
            raise ValueError('masked_gs_hard and feature_mask_hard are not found')
        masked_pred_label = \
            explainer.model.custom_forward(explainer.get_custom_input_handle_fn(
                node_explanation.masked_gs_hard,
                node_explanation.feature_mask_hard))[
                explainer.mapping_node_id()]
        node_explanation.masked_pred_label = masked_pred_label

    if "perturbed_feature_pred_label" not in node_explanation:
        if "perturbed_feature" not in node_explanation:
            perturbed_feature = generate_perturbed_feature_mask(
                explainer.feature_mask_for_output,
                explainer.config['top_k_for_faith_feature'])
            node_explanation.perturbed_feature = perturbed_feature
        perturbed_feature_pred_label = \
            explainer.model.custom_forward(explainer.get_custom_input_handle_fn(
                node_explanation.masked_gs_hard,
                node_explanation.perturbed_feature))[
                explainer.mapping_node_id()]
        node_explanation.perturbed_feature_pred_label = perturbed_feature_pred_label

    return node_explanation


def generate_perturbed_edge_mask(edge_mask, k):
    edge_mask = edge_mask[0]
    top_k = int(k * len(edge_mask))
    top_k_indices = torch.argsort(edge_mask, descending=True)[top_k:]
    perturbed_edge_mask = torch.zeros_like(edge_mask)
    perturbed_edge_mask[top_k_indices] = 1
    return perturbed_edge_mask


def graph_exp_faith_edge_explanation(node_explanation, explainer):
    """
    Set the masked_pred_label and perturbed_edge_pred_label for the faithfulness score

    :param node_explanation: a NodeExplanation object
    :param explainer: an Explainer object
    """
    if "masked_pred_label" not in node_explanation:
        flag = True
        if "masked_gs_hard" not in node_explanation and getattr(explainer,
                                                                'edge_mask_for_output',
                                                                None) is not None:
            masked_gs_hard = get_masked_gs_hard(explainer)
            node_explanation.masked_gs_hard = masked_gs_hard
            flag = False
        elif "masked_gs_hard" in node_explanation:
            flag = False
        else:
            node_explanation.masked_gs_hard = None
        if "feature_mask_hard" not in node_explanation and getattr(explainer,
                                                                   'feature_mask_for_output',
                                                                   None) is not None:
            feature_mask_hard = get_feature_mask_hard(explainer)
            node_explanation.feature_mask_hard = feature_mask_hard
            flag = False
        elif "feature_mask_hard" in node_explanation:
            flag = False
        else:
            node_explanation.feature_mask_hard = None
        if flag:
            raise ValueError('masked_gs_hard and feature_mask_hard are not found')
        masked_pred_label = \
            explainer.model.custom_forward(explainer.get_custom_input_handle_fn(
                node_explanation.masked_gs_hard,
                node_explanation.feature_mask_hard))[
                explainer.mapping_node_id()]
        node_explanation.masked_pred_label = masked_pred_label

    if "perturbed_edge_pred_label" not in node_explanation:
        if "perturbed_edge" not in node_explanation:
            perturbed_edge = generate_perturbed_edge_mask(
                explainer.edge_mask_for_output,
                explainer.config['top_k_for_faith_edge'])
            node_explanation.perturbed_edge = perturbed_edge
        perturbed_edge_pred_label = \
            explainer.model.custom_forward(explainer.get_custom_input_handle_fn(
                node_explanation.masked_gs_hard,
                node_explanation.feature_mask_hard))[
                explainer.mapping_node_id()]
        node_explanation.perturbed_edge_pred_label = perturbed_edge_pred_label

    return node_explanation


def generate_perturbed_gs(gs, perturb_ratio=0.001, neighbors_all=None):
    new_gs = []
    if neighbors_all is None:
        neighbors_all = []
        for g in gs:
            g = g.coalesce()
            edge_index = g.indices().clone()

            neighbors = {}
            for i in range(edge_index.size(1)):
                if edge_index[0, i].item() not in neighbors:
                    neighbors[edge_index[0, i].item()] = set()
                neighbors[edge_index[0, i].item()].add(edge_index[1, i].item())
            neighbors_all.append(neighbors)

    for idx, g in enumerate(gs):
        g = g.coalesce()
        edge_index = g.indices().clone()

        neighbors = neighbors_all[idx]

        def check_edge(i, j):
            return i in neighbors and j in neighbors[i]

        swap_times = round(edge_index.size(1) * perturb_ratio)
        swap_count = 0
        while swap_count < swap_times:
            i = torch.randint(0, edge_index.size(1), (1,)).item()
            j = torch.randint(0, edge_index.size(1), (1,)).item()
            if i == j:  # avoid the same index
                continue

            # avoid the same target node
            if edge_index[0, i] == edge_index[0, j] or edge_index[1, i] == edge_index[
                1, j]:
                continue

            # avoid existing edge
            # (edge_index[0, i], edge_index[1, j]) or (edge_index[0, j], edge_index[1, i]) is an existing edge
            if check_edge(edge_index[0, i].item(),
                          edge_index[1, j].item()) or check_edge(
                edge_index[0, j].item(), edge_index[1, i].item()):
                continue

            edge_index[0, i], edge_index[0, j] = edge_index[0, j], edge_index[0, i]
            swap_count += 1

        new_g = torch.sparse_coo_tensor(edge_index, g.values(), g.size())
        new_g = new_g.coalesce()
        new_gs.append(new_g)
    return new_gs, neighbors_all


def generate_perturbed_features(features, node_id, std=0.01):
    features = features.clone().detach()
    features[node_id] += torch.normal(0, std, size=(features.size(1),),
                                      device=features.device)
    return features


def generate_perturbed_mask_for_stability(explainer, neighbors=None):
    import copy
    config = copy.deepcopy(explainer.config)
    config["eval_metrics"] = []
    config["summary_path"] = None
    config["record_step"] = config["epochs"] - 1
    new_explainer = explainer.__class__(config)
    new_model = copy.deepcopy(explainer.model)
    gs, features = new_model.standard_input()
    pert_gs, neighbors = generate_perturbed_gs(gs, explainer.config.get(
        'perturb_ratio_in_gs_for_stability', 0.001), neighbors)
    pert_features = generate_perturbed_features(features, explainer.node_id,
                                                std=explainer.config.get(
                                                    'perturb_std_in_features_for_stability',
                                                    0.01))

    def standard_input(self):
        return pert_gs, pert_features

    import types
    new_model.standard_input = types.MethodType(standard_input, new_model)
    new_explainer.explain(new_model, node_id=explainer.node_id)
    if getattr(new_explainer, 'feature_mask_for_output', None) is not None:
        perturbed_feature_mask = new_explainer.feature_mask_for_output
    else:
        perturbed_feature_mask = None
    perturbed_edge_mask = new_explainer.edge_mask_for_output

    return perturbed_feature_mask, perturbed_edge_mask, neighbors


def graph_exp_stability_feature_explanation(node_explanation, explainer):
    """
    :param node_explanation: a NodeExplanation object
    :param explainer: an Explainer object
    """
    if "feature_mask" not in node_explanation:
        feature_mask = explainer.feature_mask_for_output
        node_explanation.feature_mask = feature_mask

    if "perturbed_feature_mask_for_stability" not in node_explanation:
        perturbed_feature_masks = []
        perturbed_edge_masks = []
        neighbors = None
        for time in range(explainer.config['stability_times']):
            perturbed_feature_mask, perturbed_edge_mask, neighbors = generate_perturbed_mask_for_stability(
                explainer, neighbors)
            perturbed_feature_masks.append(perturbed_feature_mask)
            perturbed_edge_masks.append(perturbed_edge_mask)
        node_explanation.perturbed_feature_mask_for_stability = perturbed_feature_masks
        node_explanation.perturbed_edge_mask_for_stability = perturbed_edge_masks

    return node_explanation


def graph_exp_stability_edge_explanation(node_explanation, explainer):
    """
    :param node_explanation: a NodeExplanation object
    :param explainer: an Explainer object
    """
    if "edge_mask" not in node_explanation:
        edge_mask = explainer.edge_mask_for_output
        node_explanation.edge_mask = edge_mask

    if "perturbed_edge_mask_for_stability" not in node_explanation:
        perturbed_feature_masks = []
        perturbed_edge_masks = []
        neighbors = None
        for time in range(explainer.config['stability_times']):
            perturbed_feature_mask, perturbed_edge_mask, neighbors = generate_perturbed_mask_for_stability(
                explainer, neighbors)
            perturbed_feature_masks.append(perturbed_feature_mask)
            perturbed_edge_masks.append(perturbed_edge_mask)
        node_explanation.perturbed_feature_mask_for_stability = perturbed_feature_masks
        node_explanation.perturbed_edge_mask_for_stability = perturbed_edge_masks

    return node_explanation


def MacroF1_explanation(node_explanation, explainer):
    """
    Set the masked_pred_label and pred_label for the MacroF1 score
    :param node_explanation: a NodeExplanation object
    :param explainer: an Explainer object
    """
    if "masked_pred_label" not in node_explanation:
        flag = True
        if "masked_gs_hard" not in node_explanation and getattr(explainer,
                                                                'edge_mask_for_output',
                                                                None) is not None:
            masked_gs_hard = get_masked_gs_hard(explainer)
            node_explanation.masked_gs_hard = masked_gs_hard
            flag = False
        elif "masked_gs_hard" in node_explanation:
            flag = False
        else:
            node_explanation.masked_gs_hard = None
        if "feature_mask_hard" not in node_explanation and getattr(explainer,
                                                                   'feature_mask_for_output',
                                                                   None) is not None:
            feature_mask_hard = get_feature_mask_hard(explainer)
            node_explanation.feature_mask_hard = feature_mask_hard
            flag = False
        elif "feature_mask_hard" in node_explanation:
            flag = False
        else:
            node_explanation.feature_mask_hard = None
        if flag:
            raise ValueError('masked_gs_hard and feature_mask_hard are not found')
        masked_pred_label = \
            explainer.model.custom_forward(explainer.get_custom_input_handle_fn(
                node_explanation.masked_gs_hard,
                node_explanation.feature_mask_hard))[
                explainer.mapping_node_id()]
        node_explanation.masked_pred_label = masked_pred_label

    if "pred_label" not in node_explanation:
        pred_label = explainer.model()[explainer.node_id]
        node_explanation.pred_label = pred_label

    return node_explanation


def MicroF1_explanation(node_explanation, explainer):
    """
    Set the masked_pred_label and pred_label for the MicroF1 score
    :param node_explanation: a NodeExplanation object
    :param explainer: an Explainer object
    """
    if "masked_pred_label" not in node_explanation:
        flag = True
        if "masked_gs_hard" not in node_explanation and getattr(explainer,
                                                                'edge_mask_for_output',
                                                                None) is not None:
            masked_gs_hard = get_masked_gs_hard(explainer)
            node_explanation.masked_gs_hard = masked_gs_hard
            flag = False
        elif "masked_gs_hard" in node_explanation:
            flag = False
        else:
            node_explanation.masked_gs_hard = None
        if "feature_mask_hard" not in node_explanation and getattr(explainer,
                                                                   'feature_mask_for_output',
                                                                   None) is not None:
            feature_mask_hard = get_feature_mask_hard(explainer)
            node_explanation.feature_mask_hard = feature_mask_hard
            flag = False
        elif "feature_mask_hard" in node_explanation:
            flag = False
        else:
            node_explanation.feature_mask_hard = None
        if flag:
            raise ValueError('masked_gs_hard and feature_mask_hard are not found')
        masked_pred_label = \
            explainer.model.custom_forward(explainer.get_custom_input_handle_fn(
                node_explanation.masked_gs_hard,
                node_explanation.feature_mask_hard))[
                explainer.mapping_node_id()]
        node_explanation.masked_pred_label = masked_pred_label

    if "pred_label" not in node_explanation:
        pred_label = explainer.model()[explainer.node_id]
        node_explanation.pred_label = pred_label

    return node_explanation


def roc_auc_score_explanation(node_explanation, explainer):
    """
    Set the masked_pred_label and pred_label for the ROC-AUC score
    :param node_explanation: a NodeExplanation object
    :param explainer: an Explainer object
    """
    if "masked_pred_label" not in node_explanation:
        flag = True
        if "masked_gs_hard" not in node_explanation and getattr(explainer,
                                                                'edge_mask_for_output',
                                                                None) is not None:
            masked_gs_hard = get_masked_gs_hard(explainer)
            node_explanation.masked_gs_hard = masked_gs_hard
            flag = False
        elif "masked_gs_hard" in node_explanation:
            flag = False
        else:
            node_explanation.masked_gs_hard = None
        if "feature_mask_hard" not in node_explanation and getattr(explainer,
                                                                   'feature_mask_for_output',
                                                                   None) is not None:
            feature_mask_hard = get_feature_mask_hard(explainer)
            node_explanation.feature_mask_hard = feature_mask_hard
            flag = False
        elif "feature_mask_hard" in node_explanation:
            flag = False
        else:
            node_explanation.feature_mask_hard = None
        if flag:
            raise ValueError('masked_gs_hard and feature_mask_hard are not found')
        masked_pred_label = \
            explainer.model.custom_forward(explainer.get_custom_input_handle_fn(
                node_explanation.masked_gs_hard,
                node_explanation.feature_mask_hard))[
                explainer.mapping_node_id()]
        node_explanation.masked_pred_label = masked_pred_label

    if "pred_label" not in node_explanation:
        pred_label = explainer.model()[explainer.node_id]
        node_explanation.pred_label = pred_label

    return node_explanation


prepare_explanation_fn_for_node_dataset_scores = {
    'fidelity_neg': fidelity_neg_explanation,
    'fidelity_pos': fidelity_pos_explanation,
    'characterization_score': characterization_score_explanation,
    'fidelity_curve_auc': fidelity_curve_auc_explanation,
    'unfaithfulness': unfaithfulness_explanation,
    'sparsity': sparsity_explanation,
    'graph_exp_faith_feature': graph_exp_faith_feature_explanation,
    'graph_exp_faith_edge': graph_exp_faith_edge_explanation,
    'graph_exp_stability_feature': graph_exp_stability_feature_explanation,
    'graph_exp_stability_edge': graph_exp_stability_edge_explanation,
    'Macro-F1': MacroF1_explanation,
    'Micro-F1': MicroF1_explanation,
    'roc_auc_score': roc_auc_score_explanation,
    'fidelity_neg_model': fidelity_neg_explanation,
    'fidelity_pos_model': fidelity_pos_explanation,
}
