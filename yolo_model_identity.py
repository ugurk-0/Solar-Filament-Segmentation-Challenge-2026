"""Verify the model that was loaded, rather than trusting an experiment name."""


def model_identity(model, expected_scale):
    network = model.model
    scale = network.yaml.get('scale')
    if model.task != 'segment' or scale != expected_scale:
        raise ValueError(f'Expected YOLO11{expected_scale}-seg; loaded task={model.task}, scale={scale}')
    return dict(task=model.task, scale=scale,
                parameters=sum(parameter.numel() for parameter in network.parameters()),
                yaml_file=str(network.yaml.get('yaml_file', '')))
