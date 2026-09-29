import numpy as np

from experiment_annotation_medoid import choose_records


def test_medoid_uses_only_supplied_training_records_and_actual_annotations(monkeypatch):
    class Coco:
        imgs = {i: {"file_name": f"{i}-observation.jpeg", "height": 8, "width": 8}
                for i in ("a", "b", "c", "excluded")}
        def getAnnIds(self, imgIds):
            assert "excluded" not in imgIds
            return imgIds
        def loadAnns(self, ids):
            return [{"id": ids[0]}]
    common = np.zeros((8, 8), bool)
    common[1:3, 1:3] = True
    outlier = np.zeros((8, 8), bool)
    outlier[5:7, 5:7] = True
    monkeypatch.setattr("experiment_annotation_medoid.sol.polygon_to_mask",
                        lambda ann, h, w: outlier if ann["id"] == "c" else common)
    selected, choices = choose_records(Coco(), ["c", "b", "a"], np.ones((8, 8), bool))
    assert selected == ["a"]
    assert choices[0]["mean_pair_pq"] == [0.5, 0.5, 0.0]
    assert choices[0]["candidates"] == ["a", "b", "c"]
