import numpy as np
import torch

from instance_embeddings import discriminative_embedding_loss, embedding_to_instances


def test_embedding_decoder_keeps_touching_instances_separate():
    height, width = 12, 20
    probability = np.ones((height, width), np.float32)
    limb = np.ones_like(probability, bool)
    embeddings = np.zeros((2, height, width), np.float32)
    embeddings[:, :, :10] = np.array([[[0.0]], [[0.0]]])
    embeddings[:, :, 10:] = np.array([[[2.0]], [[0.0]]])
    instances = embedding_to_instances(embeddings, probability, limb,
                                       threshold=0.5, min_area=10,
                                       eps=0.2, min_samples=2)
    assert len(instances) == 2
    assert sorted(int(mask.sum()) for mask in instances) == [120, 120]


def test_embedding_loss_pulls_fragments_and_pushes_instances():
    labels = torch.zeros((1, 4, 8), dtype=torch.long)
    labels[:, 1:3, :4] = 1
    labels[:, 1:3, 4:] = 2
    valid = torch.ones_like(labels, dtype=torch.bool)
    embeddings = torch.zeros((1, 2, 4, 8), requires_grad=True)
    embeddings.data[:, 0, 1:3, 4:] = 2
    loss = discriminative_embedding_loss(embeddings, labels, valid,
                                         delta_var=0.2, delta_dist=0.8)
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    assert torch.isfinite(embeddings.grad).all()


def test_embedding_decoder_ignores_low_probability_pixels():
    probability = np.zeros((8, 8), np.float32)
    probability[2:6, 2:6] = 1
    instances = embedding_to_instances(np.zeros((2, 8, 8), np.float32),
                                       probability, np.ones_like(probability, bool),
                                       threshold=0.5, min_area=1, eps=0.2,
                                       min_samples=2)
    assert len(instances) == 1
    assert int(instances[0].sum()) == 16