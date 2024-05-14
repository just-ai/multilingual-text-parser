import json

from pathlib import Path

from transformers import AutoModel

__all__ = ["load_transformer_model"]


def load_transformer_model(model_dir: Path, **kwargs):
    if model_dir.is_dir():
        if not list(model_dir.glob("*.bin")):
            config_path = model_dir / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            model_name = config["_name_or_path"]
        else:
            model_name = model_dir
    else:
        raise NotImplementedError

    return AutoModel.from_pretrained(model_name, **kwargs)
