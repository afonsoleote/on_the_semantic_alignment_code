import numpy as np

def alignment(y_hat, y_hat_i):
    diff = y_hat - y_hat_i
    return round(diff.mean(), 4)

def alignment_abs(y_hat, y_hat_i):
    return round(np.abs(y_hat - y_hat_i).mean(), 4)

def causal_sensitivity_abs(y_hat, y_hat_i, y_gt, filtered=False):

    if not filtered:
        return round(np.abs(y_hat - y_hat_i).mean(), 4)


    mask = np.round(y_hat,0) == y_gt
    return round(np.abs(y_hat[mask] - y_hat_i[mask]).mean(), 4)


def causal_sensitivity(y_hat, y_hat_q, y_gt=None, filtered=False):
    diff = y_hat - y_hat_q

    if not filtered:
        return round(diff.mean(), 4)

    mask = np.round(y_hat, 0) == y_gt
    return round(diff[mask].mean(), 4)


def counterfactual_accuracy(y_hat, y_hat_i, y_gt, y_gt_i, threshold=0.5):
    labels = np.concatenate([y_gt, y_gt_i])

    preds = np.concatenate([
        (y_hat >= threshold).astype(int),
        (y_hat_i >= threshold).astype(int)
    ])

    return round((preds == labels).mean(), 4)


def coverage_reliability_curve(preds, preds_cf, n_thresholds=100):
    deltas = preds - preds_cf

    thresholds = np.linspace(0, 1, n_thresholds)
    coverages = np.empty(n_thresholds)
    reliabilities = np.empty(n_thresholds)
    counts = np.empty(n_thresholds, dtype=int)

    n_total = len(preds)

    for i, t in enumerate(thresholds):
        mask = preds > t
        counts[i] = mask.sum()
        coverages[i] = counts[i] / n_total

        if counts[i] > 0:
            reliabilities[i] = deltas[mask].mean()
        else:
            reliabilities[i] = np.nan

    return thresholds, coverages, reliabilities, counts