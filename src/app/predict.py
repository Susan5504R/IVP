"""Predict an MDD probability from one preprocessed EEG epoch.

This is a research prototype, not a clinically validated diagnostic system.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from src.models.cdl_mddnet import CDLMDDNet


def _read_epoch(epoch_path):
    try:
        epoch = np.load(epoch_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not read epoch file '{epoch_path}': {exc}") from exc
    if epoch.ndim != 2 or epoch.shape[0] < 1 or epoch.shape[1] != 1000:
        raise ValueError(f"Expected epoch shape (channels, 1000), got {epoch.shape}")
    if (
        not np.issubdtype(epoch.dtype, np.number)
        or not np.isrealobj(epoch)
        or not np.isfinite(epoch).all()
    ):
        raise ValueError("Epoch values must be finite numbers")
    epoch = epoch.astype(np.float32, copy=False)
    if not np.isfinite(epoch).all():
        raise ValueError("Epoch values must be representable as finite float32 numbers")
    return epoch


def _read_normalizer(stats_path):
    required = {"mean", "std", "n_channels", "eps"}
    try:
        with np.load(stats_path, allow_pickle=False) as stats:
            missing = required - set(stats.files)
            if missing:
                raise ValueError(f"Normalizer is missing fields: {sorted(missing)}")
            mean = np.asarray(stats["mean"])
            std = np.asarray(stats["std"])
            channels_value = np.asarray(stats["n_channels"])
            eps_value = np.asarray(stats["eps"])
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"Could not read normalizer file '{stats_path}': {exc}") from exc

    if channels_value.shape != () or not np.issubdtype(channels_value.dtype, np.integer):
        raise ValueError("Normalizer n_channels must be an integer scalar")
    channels = int(channels_value)
    if channels < 1:
        raise ValueError("Normalizer n_channels must be positive")
    allowed_shapes = {(channels, 1), (1, channels, 1)}
    if mean.shape != std.shape or mean.shape not in allowed_shapes:
        raise ValueError(
            f"Normalizer mean/std must have matching shape (channels, 1) or "
            f"(1, channels, 1); got {mean.shape} and {std.shape}"
        )
    if (
        not np.issubdtype(mean.dtype, np.number)
        or not np.isrealobj(mean)
        or not np.isfinite(mean).all()
    ):
        raise ValueError("Normalizer mean must contain finite numbers")
    if (
        not np.issubdtype(std.dtype, np.number)
        or not np.isrealobj(std)
        or not np.isfinite(std).all()
        or (std <= 0).any()
    ):
        raise ValueError("Normalizer std must contain finite positive numbers")
    if (
        eps_value.shape != ()
        or not np.issubdtype(eps_value.dtype, np.number)
        or not np.isrealobj(eps_value)
    ):
        raise ValueError("Normalizer eps must be a numeric scalar")
    eps = float(eps_value)
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("Normalizer eps must be finite and positive")
    mean = mean.astype(np.float32)
    std = std.astype(np.float32)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Normalizer statistics must be representable as finite float32 values")
    return mean, std, channels, eps


def predict_epoch(epoch_path, checkpoint_path, stats_path, device="cpu"):
    epoch_path = Path(epoch_path)
    checkpoint_path = Path(checkpoint_path)
    stats_path = Path(stats_path)
    for path, label in (
        (epoch_path, "Epoch"),
        (checkpoint_path, "Inference checkpoint"),
        (stats_path, "Normalizer"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} file not found: '{path}'")

    epoch = _read_epoch(epoch_path)
    mean, std, channels, eps = _read_normalizer(stats_path)
    if epoch.shape[0] != channels:
        raise ValueError(
            f"Normalizer expects {channels} channels, got {epoch.shape[0]}"
        )

    try:
        target_device = torch.device(device)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"Invalid device '{device}'") from exc
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")

    normalized = (epoch[None] - mean) / np.maximum(std, eps)
    model = CDLMDDNet(n_channels=channels, n_samples=epoch.shape[1], n_classes=2)
    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(f"Could not load inference checkpoint '{checkpoint_path}': {exc}") from exc
    if not isinstance(state, dict) or not state or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state.items()
    ):
        raise ValueError(
            "Expected model.pt inference checkpoint to contain a state_dict; "
            "training-resume checkpoints are not supported"
        )
    try:
        model.load_state_dict(state, strict=True)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"Checkpoint is incompatible with CDL-MDDNet: {exc}") from exc

    model.to(target_device).eval()
    tensor = torch.from_numpy(normalized.astype(np.float32, copy=False)).to(target_device)
    with torch.inference_mode():
        probability = float(torch.softmax(model(tensor), dim=1)[0, 1].item())
    if not np.isfinite(probability):
        raise RuntimeError("Model produced a non-finite MDD probability")
    return {
        "prediction": "MDD" if probability >= 0.5 else "HC",
        "p_mdd": probability,
        "threshold": 0.5,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epoch", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalizer", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    try:
        result = predict_epoch(args.epoch, args.checkpoint, args.normalizer, args.device)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(
        f"Prediction: {result['prediction']} | P(MDD): {result['p_mdd']:.4f} "
        f"| threshold: {result['threshold']:.1f}"
    )
    print("Research prototype only; not for clinical use.")


if __name__ == "__main__":
    main()
