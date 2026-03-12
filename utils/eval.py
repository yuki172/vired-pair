import torch


def _binary_average_precision(y_true: torch.Tensor, y_score: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Average Precision (AP) for binary classification.
    y_true: (N,) in {0,1}
    y_score: (N,) probability or score for positive class
    """
    y_true = y_true.float()
    y_score = y_score.float()

    n_pos = y_true.sum()
    if n_pos == 0:
        return 0.0

    order = torch.argsort(y_score, descending=True)
    y_true = y_true[order]

    tp = torch.cumsum(y_true, dim=0)
    fp = torch.cumsum(1.0 - y_true, dim=0)

    precision = tp / (tp + fp + eps)
    recall = tp / (n_pos + eps)

    # AP = sum over recall increments of precision
    recall_prev = torch.cat([torch.zeros(1, device=recall.device), recall[:-1]])
    ap = torch.sum((recall - recall_prev) * precision)
    return float(ap.item())


def _binary_auroc(y_true: torch.Tensor, y_score: torch.Tensor, eps: float = 1e-8) -> float:
    """
    AUROC for binary classification using rank statistics.
    y_true: (N,) in {0,1}
    y_score: (N,) probability or score for positive class
    """
    y_true = y_true.long()
    y_score = y_score.float()

    n_pos = (y_true == 1).sum().item()
    n_neg = (y_true == 0).sum().item()

    if n_pos == 0 or n_neg == 0:
        return 0.0

    order = torch.argsort(y_score)
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, len(y_score) + 1, device=y_score.device, dtype=torch.float)

    pos_ranks_sum = ranks[y_true == 1].sum()
    auc = (pos_ranks_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg + eps)
    return float(auc.item())


def pair_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    padding_mask: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-8,
):
    """
    Args:
        logits: (B, P_max, 2)
        labels: (B, P_max) with values in {0,1}
        padding_mask: (B, P_max), True means padding and should be ignored
        threshold: decision threshold for positive class
        eps: numerical stability

    Returns:
        dict with:
            - accuracy
            - precision
            - recall
            - f1
            - ap
            - auroc
            - num_valid
    """
    if logits.ndim != 3 or logits.shape[-1] != 2:
        raise ValueError(f"logits must have shape (B, P_max, 2), got {tuple(logits.shape)}")
    if labels.ndim != 2:
        raise ValueError(f"labels must have shape (B, P_max), got {tuple(labels.shape)}")
    if padding_mask.ndim != 2:
        raise ValueError(f"padding_mask must have shape (B, P_max), got {tuple(padding_mask.shape)}")
    if logits.shape[:2] != labels.shape or labels.shape != padding_mask.shape:
        raise ValueError(
            f"Shape mismatch: logits.shape[:2]={tuple(logits.shape[:2])}, "
            f"labels.shape={tuple(labels.shape)}, padding_mask.shape={tuple(padding_mask.shape)}"
        )

    valid_mask = ~padding_mask
    num_valid = int(valid_mask.sum().item())

    if num_valid == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "ap": 0.0,
            "auroc": 0.0,
            "num_valid": 0,
        }

    probs = torch.softmax(logits, dim=-1)[..., 1]
    preds = (probs >= threshold).long()

    y_true = labels[valid_mask].long()
    y_score = probs[valid_mask]
    y_pred = preds[valid_mask].long()

    tp = ((y_pred == 1) & (y_true == 1)).sum().item()
    tn = ((y_pred == 0) & (y_true == 0)).sum().item()
    fp = ((y_pred == 1) & (y_true == 0)).sum().item()
    fn = ((y_pred == 0) & (y_true == 1)).sum().item()

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, eps)
    recall = tp / max(tp + fn, eps)
    f1 = 2 * precision * recall / max(precision + recall, eps)

    ap = _binary_average_precision(y_true, y_score, eps=eps)
    auroc = _binary_auroc(y_true, y_score, eps=eps)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "ap": float(ap),
        "auroc": float(auroc),
        "num_valid": num_valid,
    }