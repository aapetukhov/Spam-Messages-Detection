from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay

import matplotlib.pyplot as plt
import numpy as np

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