"""Shared configuration for inspection and training data loaders."""
from functools import partial
from pathlib import Path

import yaml
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/stage2_data_synth100k.yaml"


def load_data_config(path=DEFAULT_CONFIG, *, project_root=PROJECT_ROOT):
    with open(path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    dataset = config["dataset"]
    loader = config["dataloader"]
    for name in ("image_height", "max_width"):
        value = dataset.get(name)
        if name == "max_width" and value is None:
            continue
        if type(value) is not int or value <= 0:
            raise ValueError(f"dataset.{name} must be a positive integer")
    for name, minimum in (("batch_size", 1), ("num_workers", 0)):
        value = loader[name]
        if type(value) is not int or value < minimum:
            raise ValueError(f"dataloader.{name} must be >= {minimum}")
    if type(dataset["invert_image"]) is not bool:
        raise ValueError("dataset.invert_image must be true or false")
    if dataset.get("augment", False):
        raise ValueError("Augmentation is not implemented in Stage 2.3")
    if dataset.get("pixel_range", [0.0, 1.0]) != [0.0, 1.0]:
        raise ValueError("Only pixel_range [0.0, 1.0] is supported")
    config["paths"] = {
        key: str(Path(project_root) / value) for key, value in config["paths"].items()
    }
    return config


def dataset_options(config):
    settings = config["dataset"]
    return {key: settings[key] for key in
            ("image_height", "max_width", "augment", "invert_image")}


def make_dataloader(dataset, config, **overrides):
    """Keep padding, worker settings, and pixel inversion in agreement."""
    from stage2_htr.collate import htr_collate_fn

    options = dict(config["dataloader"])
    options.update(overrides)
    underlying = dataset
    while isinstance(underlying, Subset):
        underlying = underlying.dataset
    invert = getattr(underlying, "invert_image", config["dataset"]["invert_image"])
    return DataLoader(
        dataset, collate_fn=partial(
            htr_collate_fn, pad_value=0.0 if invert else 1.0
        ), **options,
    )
