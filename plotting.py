from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import clear_output
from typing import Tuple
import numpy as np
import torch

def depict_pr_roc(y_true, y_pred, classifier_name='Some Classifier', ax=None):
    """
    Generate precision-recall and ROC curves for a classifier, displaying AUC values.

    Parameters:
    - y_true: array-like of shape (n_samples,) or (n_samples, n_classes)
        True binary labels or probabilities of the positive class.
    - y_pred: array-like of shape (n_samples,) or (n_samples, n_classes)
        Target scores, can be probabilities or decision function.
    - classifier_name: str, default='Some Classifier'
        Name of the classifier (for display purposes).
    - ax: object, default=None
        Axes to plot the curves. If None, a new figure will be created.

    Returns:
    None
    """
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(11, 5))

    print(classifier_name, 'metrics')
    PrecisionRecallDisplay.from_predictions(y_true, y_pred, ax=ax[0], name=classifier_name)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    print('AUC-PR: %.5f' % auc(recall, precision))
    ax[0].set_title("PRC")
    ax[0].set_ylim(0, 1.1)

    RocCurveDisplay.from_predictions(y_true, y_pred, ax=ax[1], name=classifier_name)
    print('AUC-ROC: %.5f' % roc_auc_score(y_true, y_pred))
    ax[1].set_title("ROC")
    ax[1].set_ylim(0, 1.1)

    plt.tight_layout()
    plt.legend()

def plot_calibration_curve(y_test, preds, classifier_name):
    """
    Generate a calibration curve to assess the predictive accuracy of a binary classifier.

    Parameters:
    - y_test (array-like): True binary labels of the test set.
    - preds (array-like): Predicted probabilities of the positive class in the test set.
    - classifier_name (str): Name of the classifier.

    Returns:
    None
    """
    bin_middle_points = []
    bin_real_ratios = []
    n_bins = 10
    for i in range(n_bins):
        l = 1.0 / n_bins * i
        r = 1.0 / n_bins * (i + 1)
        bin_middle_points.append((l + r) / 2)
        bin_real_ratios.append(np.mean(y_test[(preds >= l) & (preds < r)] == 1))
    plt.figure(figsize=(6,6))
    plt.plot(bin_middle_points, bin_real_ratios, label = 'Calibration curve', marker = '.')
    plt.plot([0, 1], [0, 1], 'k--', label = 'perfect')
    plt.title(f'Calibration curve for {classifier_name}', weight = 'bold', fontsize = 12)
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.legend(fontsize = 12)
    plt.ylim([-0.05, 1.05])
    plt.grid()

def plot_losses(train_losses, test_losses, train_accuracies, test_accuracies):
    clear_output()
    def to_cpu_numpy(data):
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return data
    
    train_losses = [to_cpu_numpy(loss) for loss in train_losses]
    test_losses = [to_cpu_numpy(loss) for loss in test_losses]
    train_accuracies = [to_cpu_numpy(acc) for acc in train_accuracies]
    test_accuracies = [to_cpu_numpy(acc) for acc in test_accuracies]

    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    axs[0].plot(range(1, len(train_losses) + 1), train_losses, label='train')
    axs[0].plot(range(1, len(test_losses) + 1), test_losses, label='test')
    axs[0].set_ylabel('loss')

    axs[1].plot(range(1, len(train_accuracies) + 1), train_accuracies, label='train')
    axs[1].plot(range(1, len(test_accuracies) + 1), test_accuracies, label='test')
    axs[1].set_ylabel('accuracy')

    for ax in axs:
        ax.set_xlabel('epoch')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.show()

def attention_heatmap(texts: list, sentence_num: int, attention_scores: np.ndarray, figsize: Tuple[int, int]=(14, 6)):
    """
    Visualises attention scores for 2 arbitrary attention heads (matrices have to be already preprocessed)
    """
    text = texts[sentence_num].split()
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    sns.heatmap(np.round(attention_scores[0].cpu().numpy(), 2), ax=ax[0], cmap='Blues', annot=True, linewidths=.5, cbar=False, xticklabels=text, yticklabels=text, linecolor='black')
    sns.heatmap(np.round(attention_scores[1].cpu().numpy(), 2), ax=ax[1], cmap='Blues', annot=True, linewidths=.5, cbar=False, xticklabels=text, yticklabels=text, linecolor='black')

    plt.suptitle(f'Визуализация механизма внимания', weight='bold')
    ax[0].set_title(f'Head 1', weight='bold')
    ax[1].set_title(f'Head 2', weight='bold')
    for i in range(2):
        ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=60)
    
    plt.show()

def plot_metrics(train_losses, test_losses, train_metrics, test_metrics):
    fig = plt.figure(figsize=(12, 6))
    ax = plt.gca()
    X_axis = range(1, len(train_losses) + 1)
    results = {
        'train_BCE': train_losses,
        'train_AUC-ROC': train_metrics,
        'test_BCE': test_losses,
        'test_AUC-ROC': test_metrics        
    }

    for scorer, color in zip(['BCE', 'AUC-ROC'], ["green", "black"]):
        for sample_part, style in (("train", "--"), ("test", "-")):
            sample_score_mean = results[f"{sample_part}_{scorer}"]
            ax.plot(
                X_axis,
                sample_score_mean,
                style,
                color=color,
                alpha=1 if sample_part == "test" else 0.7,
                label=f"{scorer} ({sample_part})"
            )

        best_index = np.nonzero(results[f"rank_test_{scorer}"] == 1)[0][0]
        best_score = results[f"mean_test_{scorer}"][best_index]

        # отмечаем максимумы
        ax.scatter(
            [X_axis[best_index]],
            [best_score],
            color=color,
            marker="o",
        )

        ax.annotate(f"{best_score:.2f}", (X_axis[best_index], best_score + 0.01), weight='bold')

    plt.title('Train & Test losses and metrics', weight='bold', fontsize=12)
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True, shadow=True)
    plt.grid(alpha=0.4, linestyle='--')

    plt.show()