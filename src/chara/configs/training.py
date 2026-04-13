from dataclasses import dataclass

import torch


@dataclass
class TrainingConfig:
    """Training settings and options"""

    batch_size: int
    """number of samples or sequences per optimization step"""
    epochs: int
    """number of full passes over the dataset"""
    learning_rate: float
    """step size of loss correction during backward pass"""
    device: torch.types.Device
    """pytorch device for training and inference"""
