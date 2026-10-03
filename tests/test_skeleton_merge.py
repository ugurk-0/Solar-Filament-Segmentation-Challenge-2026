import numpy as np

from skeleton_merge import merge_instances, skeleton_endpoints


def test_merge_bridges_short_aligned_gap():
    left = np.zeros((16, 32), bool)
    left[8, 4:13] = True
    right = np.zeros_like(left)
    right[8, 17:28] = True
    probability = np.ones((16, 32), np.float32)
    merged = merge_instances([left, right], probability, max_distance=10)
    assert len(merged) == 1
    assert int(merged[0].sum()) == 20


def test_no_merge_when_gap_probability_low():
    left = np.zeros((16, 32), bool)
    left[8, 4:13] = True
    right = np.zeros_like(left)
    right[8, 17:28] = True
    probability = np.zeros((16, 32), np.float32)
    merged = merge_instances([left, right], probability, max_distance=10)
    assert len(merged) == 2


def test_no_merge_when_misaligned():
    top = np.zeros((32, 16), bool)
    top[4:13, 8] = True
    bottom = np.zeros_like(top)
    bottom[17:28, 8] = True
    probability = np.ones((32, 16), np.float32)
    # Perpendicular endpoints: tangent is vertical, but the merge rule requires
    # alignment with the gap direction, which is vertical here too, so this pair
    # WOULD merge. Use horizontal offset instead to test misalignment.
    offset = np.zeros_like(top)
    offset[17:28, 12] = True
    merged = merge_instances([top, offset], probability, max_distance=12,
                             min_cosine=0.8)
    assert len(merged) == 2


def test_endpoints_and_identity_cases():
    line = np.zeros((9, 9), bool)
    line[4, 2:7] = True
    endpoints = skeleton_endpoints(line)
    assert len(endpoints) == 2
    single = [line]
    assert merge_instances(single, np.ones((9, 9), np.float32)) == single
    assert merge_instances([], np.ones((9, 9), np.float32)) == []
