import numpy as np
import pytest
import torch

from src.app.predict import main as predictor_main
from src.app.predict import predict_epoch
from src.models.cdl_mddnet import CDLMDDNet


@pytest.fixture
def inference_files(tmp_path):
    channels = 4
    epoch = np.random.default_rng(42).normal(size=(channels, 1000)).astype(np.float32)
    mean = np.full((1, channels, 1), 0.25, dtype=np.float32)
    std = np.full((1, channels, 1), 2.0, dtype=np.float32)
    epoch_path = tmp_path / "epoch.npy"
    checkpoint_path = tmp_path / "model.pt"
    stats_path = tmp_path / "normalizer_stats.npz"
    np.save(epoch_path, epoch)
    np.savez(stats_path, mean=mean, std=std, eps=np.float32(1e-8), n_channels=np.int64(channels))
    model = CDLMDDNet(n_channels=channels, n_samples=1000, n_classes=2)
    torch.save(model.state_dict(), checkpoint_path)
    return epoch, epoch_path, checkpoint_path, stats_path


def test_predicts_from_matching_epoch_checkpoint_and_normalizer(inference_files):
    epoch, epoch_path, checkpoint_path, stats_path = inference_files
    result = predict_epoch(epoch_path, checkpoint_path, stats_path)

    model = CDLMDDNet(n_channels=epoch.shape[0], n_samples=1000, n_classes=2)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    model.eval()
    with torch.inference_mode():
        normalized = (epoch[None] - 0.25) / 2.0
        expected = torch.softmax(
            model(torch.from_numpy(normalized.astype(np.float32))), dim=1
        )[0, 1].item()

    assert result["p_mdd"] == pytest.approx(expected)
    assert result["prediction"] == ("MDD" if expected >= 0.5 else "HC")
    assert result["threshold"] == 0.5


@pytest.mark.parametrize("shape", [(4, 999), (4, 1000, 1), (0, 1000)])
def test_rejects_invalid_epoch_dimensions(inference_files, shape, tmp_path):
    _, _, checkpoint_path, stats_path = inference_files
    epoch_path = tmp_path / "invalid_epoch.npy"
    np.save(epoch_path, np.zeros(shape, dtype=np.float32))

    with pytest.raises(ValueError, match="Expected epoch shape"):
        predict_epoch(epoch_path, checkpoint_path, stats_path)


@pytest.mark.parametrize("missing", ["epoch", "checkpoint", "normalizer"])
def test_reports_missing_files(inference_files, missing, tmp_path):
    _, epoch_path, checkpoint_path, stats_path = inference_files
    paths = {"epoch": epoch_path, "checkpoint": checkpoint_path, "normalizer": stats_path}
    paths[missing] = tmp_path / f"missing_{missing}"

    with pytest.raises(FileNotFoundError, match="not found"):
        predict_epoch(paths["epoch"], paths["checkpoint"], paths["normalizer"])


def test_cli_reports_missing_input_without_traceback(inference_files, monkeypatch, capsys, tmp_path):
    _, _, checkpoint_path, stats_path = inference_files
    epoch_path = tmp_path / "missing_epoch.npy"
    monkeypatch.setattr(
        "sys.argv",
        [
            "predict",
            "--epoch", str(epoch_path),
            "--checkpoint", str(checkpoint_path),
            "--normalizer", str(stats_path),
        ],
    )

    with pytest.raises(SystemExit) as error:
        predictor_main()

    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert "Epoch file not found" in stderr
    assert "Traceback" not in stderr


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_rejects_invalid_epoch_values(inference_files, invalid_value, tmp_path):
    epoch, _, checkpoint_path, stats_path = inference_files
    epoch[0, 0] = invalid_value
    epoch_path = tmp_path / "invalid_values.npy"
    np.save(epoch_path, epoch)

    with pytest.raises(ValueError, match="finite numbers"):
        predict_epoch(epoch_path, checkpoint_path, stats_path)


def test_rejects_channel_mismatch(inference_files, tmp_path):
    _, _, checkpoint_path, stats_path = inference_files
    epoch_path = tmp_path / "wrong_channels.npy"
    np.save(epoch_path, np.zeros((5, 1000), dtype=np.float32))

    with pytest.raises(ValueError, match="expects 4 channels"):
        predict_epoch(epoch_path, checkpoint_path, stats_path)


def test_rejects_incompatible_checkpoint(inference_files, tmp_path):
    _, epoch_path, _, stats_path = inference_files
    checkpoint_path = tmp_path / "incompatible.pt"
    incompatible_model = CDLMDDNet(n_channels=5, n_samples=1000, n_classes=2)
    torch.save(incompatible_model.state_dict(), checkpoint_path)

    with pytest.raises(ValueError, match="incompatible with CDL-MDDNet"):
        predict_epoch(epoch_path, checkpoint_path, stats_path)


def test_rejects_training_resume_checkpoint(inference_files, tmp_path):
    _, epoch_path, _, stats_path = inference_files
    checkpoint_path = tmp_path / "training_checkpoint.pt"
    torch.save({"model_state_dict": {}}, checkpoint_path)

    with pytest.raises(ValueError, match="training-resume checkpoints"):
        predict_epoch(epoch_path, checkpoint_path, stats_path)


def test_rejects_invalid_normalizer_values(inference_files, tmp_path):
    _, epoch_path, checkpoint_path, _ = inference_files
    stats_path = tmp_path / "invalid_stats.npz"
    np.savez(
        stats_path,
        mean=np.zeros((1, 4, 1), dtype=np.float32),
        std=np.full((1, 4, 1), np.nan, dtype=np.float32),
        eps=np.float32(1e-8),
        n_channels=np.int64(4),
    )

    with pytest.raises(ValueError, match="std must contain finite positive numbers"):
        predict_epoch(epoch_path, checkpoint_path, stats_path)