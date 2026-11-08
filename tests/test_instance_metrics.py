import numpy as np
import pytest

from instance_metrics import matrix_metrics
from solution import instance_iou_matrix, panoptic_quality


@pytest.mark.parametrize("case", ["empty", "one", "merge", "split", "below"])
def test_cached_metric_matches_pipeline(case):
    a, b = np.zeros((12, 12), bool), np.zeros((12, 12), bool)
    a[1:5, 1:5] = True
    b[7:11, 7:11] = True
    split = a.copy()
    split[3:] = False
    pred, gt = {"empty": ([], []), "one": ([a], [a]), "merge": ([a | b], [a, b]),
                "split": ([split, a & ~split], [a]), "below": ([b], [a])}[case]
    actual = matrix_metrics(instance_iou_matrix(pred, gt))
    assert actual["pq"] == pytest.approx(panoptic_quality(pred, gt)[0])
