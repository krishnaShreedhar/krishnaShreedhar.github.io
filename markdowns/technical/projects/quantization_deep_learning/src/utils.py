"""
Shared utilities for quantization_deep_learning project.
Provides config loading, logging setup, model definitions, and dataset generation.
"""

import logging
import logging.handlers
import os
import pathlib
from typing import Any

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict[str, Any]:
    """Load YAML configuration file and return as nested dict."""
    with open(config_path, "r") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(cfg: dict[str, Any]) -> logging.Logger:
    """
    Configure root logger from config dict.

    Args:
        cfg: Full project config dict (uses cfg['logging'] section).

    Returns:
        Configured root logger.
    """
    log_cfg = cfg["logging"]
    level_str: str = log_cfg.get("level", "INFO")
    log_file: str = log_cfg.get("log_file", "logs/quantization_dl.log")
    max_bytes: int = log_cfg.get("max_bytes", 104857600)
    backup_count: int = log_cfg.get("backup_count", 5)

    level = getattr(logging, level_str.upper(), logging.INFO)

    pathlib.Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates on repeated calls
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Rotating file handler
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    return root_logger


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

class LeNetCNN(nn.Module):
    """
    LeNet-style CNN for synthetic MNIST-like classification.

    Architecture: Conv → BN → ReLU → Conv → BN → ReLU → AdaptivePool → Linear → Linear
    Designed to be quantization-friendly (fused BN patterns).
    """

    def __init__(self, in_channels: int, num_classes: int, hidden_dim: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        # After two 2x2 max-pools on 28x28 input: 7x7
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class QuantizableLeNetCNN(nn.Module):
    """
    Quantization-ready LeNet CNN with explicit QuantStub/DeQuantStub.
    Uses torch.nn.quantizable-compatible structure for PTQ/QAT.
    """

    def __init__(self, in_channels: int, num_classes: int, hidden_dim: int) -> None:
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=False)  # inplace=False required for QAT
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 7 * 7, hidden_dim)
        self.relu3 = nn.ReLU(inplace=False)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)

        x = self.dequant(x)
        return x

    def fuse_modules(self) -> None:
        """Fuse Conv-BN-ReLU patterns for quantization efficiency."""
        torch.quantization.fuse_modules(
            self,
            [["conv1", "bn1", "relu1"], ["conv2", "bn2", "relu2"]],
            inplace=True,
        )


class SimpleLSTM(nn.Module):
    """
    Simple LSTM model for dynamic quantization demonstration.
    Dynamic PTQ works best on weight-heavy layers like LSTM and Linear.
    """

    def __init__(self, input_size: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers=2, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        # Use last timestep
        return self.classifier(out[:, -1, :])


# ---------------------------------------------------------------------------
# Synthetic dataset generation
# ---------------------------------------------------------------------------

def generate_synthetic_dataset(
    num_samples: int,
    in_channels: int,
    height: int,
    width: int,
    num_classes: int,
    random_seed: int,
) -> TensorDataset:
    """
    Generate synthetic image-like dataset with random tensors.

    Args:
        num_samples: Number of samples to generate.
        in_channels: Number of image channels.
        height: Image height in pixels.
        width: Image width in pixels.
        num_classes: Number of output classes.
        random_seed: Random seed for reproducibility.

    Returns:
        TensorDataset of (images, labels).
    """
    generator = torch.Generator()
    generator.manual_seed(random_seed)

    images = torch.randn(num_samples, in_channels, height, width, generator=generator)
    labels = torch.randint(0, num_classes, (num_samples,), generator=generator)
    return TensorDataset(images, labels)


def build_dataloaders(
    cfg: dict[str, Any],
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, calibration, and test DataLoaders from config.

    Args:
        cfg: Full project config dict.

    Returns:
        Tuple of (train_loader, calib_loader, test_loader).
    """
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    logger = logging.getLogger(__name__)

    num_samples: int = data_cfg["num_samples"]
    num_calib: int = data_cfg["num_calibration_samples"]
    batch_size: int = data_cfg["batch_size"]
    num_workers: int = data_cfg["num_workers"]
    seed: int = data_cfg["random_seed"]

    in_channels: int = model_cfg["in_channels"]
    height: int = model_cfg["input_height"]
    width: int = model_cfg["input_width"]
    num_classes: int = model_cfg["num_classes"]

    train_size = int(0.8 * num_samples)
    test_size = num_samples - train_size

    full_dataset = generate_synthetic_dataset(
        num_samples, in_channels, height, width, num_classes, seed
    )

    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )

    # Calibration set: first num_calib samples from train
    calib_images, calib_labels = full_dataset.tensors
    calib_dataset = TensorDataset(
        calib_images[:num_calib], calib_labels[:num_calib]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    calib_loader = DataLoader(
        calib_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    logger.info(
        "DataLoaders built | train=%d, calib=%d, test=%d samples",
        train_size,
        num_calib,
        test_size,
    )
    return train_loader, calib_loader, test_loader


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run a single training epoch, return average loss."""
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def evaluate_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluate model accuracy on a DataLoader, return accuracy in [0, 1]."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


def get_model_size_mb(model: nn.Module) -> float:
    """Return model size in megabytes by counting parameter bytes."""
    total_bytes = 0
    for param in model.parameters():
        total_bytes += param.nelement() * param.element_size()
    for buf in model.buffers():
        total_bytes += buf.nelement() * buf.element_size()
    return total_bytes / (1024 ** 2)


def ensure_output_dir(cfg: dict[str, Any]) -> pathlib.Path:
    """Create and return the output directory for models."""
    out_dir = pathlib.Path(cfg["model"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
