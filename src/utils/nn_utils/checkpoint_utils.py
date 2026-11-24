import torch
from pathlib import Path

from src.utils.models.type_a_model_recover import ConvExp16Recover


def load_checkpoint(path, device="cpu"):

    checkpoint = torch.load(Path(path), map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        config = checkpoint.get("config", None)
    else:
        state_dict = checkpoint
        config = None

    return state_dict, config

def load_model(path, device="cpu"):
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)

    state_dict = checkpoint["state_dict"]
    config = {
        "model_class": checkpoint["model_class"],
        "model_kwargs": checkpoint.get("model_kwargs", {})
    }

    model_class = config["model_class"]
    model_kwargs = config.get("model_kwargs", {})

    model = model_class.build(**model_kwargs)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model