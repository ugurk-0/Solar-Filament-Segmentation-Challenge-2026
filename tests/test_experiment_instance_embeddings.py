from types import SimpleNamespace

import numpy as np
import pytest
import torch

import experiment_instance_embeddings as exp
import solution as sol


def test_build_instance_labels_separates_polygons(monkeypatch):
    coco = SimpleNamespace(
        getAnnIds=lambda imgIds: [1, 2],
        loadAnns=lambda annIds: [
            {"segmentation": [[1, 1, 3, 1, 3, 3, 1, 3]]},
            {"segmentation": [[4, 4, 6, 4, 6, 6, 4, 6]]},
        ],
    )

    def fake_polygon(annotation, height, width):
        mask = np.zeros((height, width), bool)
        if annotation["segmentation"][0][0] == 1:
            mask[1:3, 1:3] = True
        else:
            mask[4:6, 4:6] = True
        return mask

    monkeypatch.setattr(sol, "polygon_to_mask", fake_polygon)
    labels = exp.build_instance_labels(coco, 7, 8, 8)
    assert labels.dtype == np.int64
    assert set(np.unique(labels)) == {0, 1, 2}
    assert np.all(labels[1:3, 1:3] == 1)
    assert np.all(labels[4:6, 4:6] == 2)


def test_instance_dataset_keeps_labels_aligned_through_spatial_transforms(monkeypatch):
    cfg = sol.Cfg(img_size=16, crop_size=8, oversample_p=0.0)
    dataset = exp.InstanceLabelDataset(None, [3], cfg, True, np.ones((16, 16), bool))
    image = np.zeros((16, 16), np.float32)
    image[2:4, 10:12] = 1
    mask = image.astype(bool)
    labels = np.zeros((16, 16), np.int64)
    labels[2:4, 10:12] = 5
    dataset._cache[3] = (image, mask, np.zeros((16, 16), np.float32), labels)
    values = iter([0.9, 0.1, 0.9, 0.9])
    monkeypatch.setattr(exp.random, "random", lambda: next(values))
    monkeypatch.setattr(exp.random, "randint", lambda low, high: 2)
    monkeypatch.setattr(exp.random, "randrange", lambda n: 1)
    x, y, _, vm, instance_labels = dataset[0]
    assert torch.equal((x > 0.5)[0], y[0].bool())
    assert torch.equal(y[0].bool(), instance_labels.bool())
    assert torch.equal(vm[0].bool(), torch.ones((8, 8), dtype=torch.bool))
    assert int(instance_labels.max()) == 5


def test_warm_start_accepts_missing_embedding_head(tmp_path):
    cfg_parent = sol.Cfg(device="cpu", embedding_dim=0)
    parent = sol.FilamentUNet(cfg_parent, pretrained=False)
    checkpoint = tmp_path / "parent.pt"
    torch.save({"cfg": {"fold": 1}, "ema": parent.state_dict()}, checkpoint)
    cfg_child = sol.Cfg(device="cpu", fold=1, embedding_dim=4)
    child = sol.FilamentUNet(cfg_child, pretrained=False)
    report = exp.load_warm_start(child, checkpoint, fold=1)
    assert report["unexpected"] == []
    assert report["missing"] == ["head_embed.bias", "head_embed.weight"]


def test_warm_start_rejects_fold_mismatch(tmp_path):
    cfg_parent = sol.Cfg(device="cpu", embedding_dim=0)
    parent = sol.FilamentUNet(cfg_parent, pretrained=False)
    checkpoint = tmp_path / "parent.pt"
    torch.save({"cfg": {"fold": 0}, "ema": parent.state_dict()}, checkpoint)
    cfg_child = sol.Cfg(device="cpu", fold=1, embedding_dim=4)
    child = sol.FilamentUNet(cfg_child, pretrained=False)
    with pytest.raises(ValueError):
        exp.load_warm_start(child, checkpoint, fold=1)