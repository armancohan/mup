import os
from typing import Any, Dict, Optional

import torch
from pytorch_lightning.plugins import CheckpointIO


class CustomCheckpointIO(CheckpointIO):
    def save_checkpoint(self, checkpoint: Dict[str, Any], path: str, storage_options: Optional[Any] = None) -> None:
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str, storage_options: Optional[Any] = None) -> Dict[str, Any]:
        return torch.load(path)

    def remove_checkpoint(self, path: str) -> None:
        os.remove(path)
