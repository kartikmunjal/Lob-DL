import json

import numpy as np
import pytest
import torch

from lobdl.factorial import build_neural
from lobdl.features import causal_features
from lobdl.models import DeepLOB, DeepLOBFused, LSTMHead, count_parameters
from lobdl.representations import (
    ENG_CLIP,
    ENG_COLUMNS,
    RAW_LAGS,
    EngineeredScaler,
    FactorialWindowDataset,
    channel_count,
    engineered_day,
    flat_feature_names,
    flat_matrix,
)

HORIZON = 10  # events -> one representation row
SEGMENT_ROWS = 400


def _book(rng, rows):
    mid = 100.0 + np.cumsum(rng.normal(0, 0.02, rows))
    columns = np.zeros((rows, 40))
    for level in range(10):
        columns[:, 4 * level + 0] = mid + 0.05 * (level + 1)
        columns[:, 4 * level + 1] = rng.integers(10, 500, rows)
        columns[:, 4 * level + 2] = mid - 0.05 * (level + 1)
        columns[:, 4 * level + 3] = rng.integers(10, 500, rows)
    return columns


def _make_root(tmp_path, days=(1, 2, 7)):
    rng = np.random.default_rng(0)
    entries = []
    for day in range(1, 11):
        matrix = np.ones((2 * SEGMENT_ROWS, 149))
        if day in days:
            for stock in range(2):
                start = stock * SEGMENT_ROWS
                matrix[start : start + SEGMENT_ROWS, :40] = _book(rng, SEGMENT_ROWS)
                entries.append(
                    {"day": day, "stock": stock, "start": start, "stop": start + SEGMENT_ROWS}
                )
            matrix[:, 144:149] = rng.integers(1, 4, (2 * SEGMENT_ROWS, 5))
            np.save(tmp_path / f"day_{day:02d}.npy", matrix.astype(np.float32))
    (tmp_path / "segments.json").write_text(json.dumps(entries), encoding="utf-8")
    return tmp_path


def test_engineered_day_matches_per_segment_causal_features(tmp_path):
    root = _make_root(tmp_path)
    full = engineered_day(root, 1)
    matrix = np.load(root / "day_01.npy")
    for start in (0, SEGMENT_ROWS):
        expected, _ = causal_features(matrix[start : start + SEGMENT_ROWS, :40])
        assert np.allclose(full[start : start + SEGMENT_ROWS], expected)
    assert full.shape[1] == ENG_COLUMNS


def test_engineered_features_are_causal(tmp_path):
    root = _make_root(tmp_path)
    matrix = np.load(root / "day_01.npy")
    block = matrix[:SEGMENT_ROWS, :40].copy()
    baseline, _ = causal_features(block)
    cut = 250
    altered = block.copy()
    altered[cut + 1 :] = _book(np.random.default_rng(99), SEGMENT_ROWS - cut - 1)
    changed, _ = causal_features(altered)
    assert np.allclose(baseline[: cut + 1], changed[: cut + 1])


def test_scaler_uses_training_days_only_and_clips(tmp_path):
    root = _make_root(tmp_path)
    with pytest.raises(ValueError, match="training days"):
        EngineeredScaler.fit(root, days=(1, 7))
    scaler = EngineeredScaler.fit(root, days=(1, 2))
    stacked = np.vstack([engineered_day(root, d) for d in (1, 2)]).astype(np.float64)
    assert np.allclose(scaler.mean, stacked.mean(axis=0))
    spike = np.full((1, ENG_COLUMNS), 1e9)
    assert scaler.transform(spike).max() == pytest.approx(ENG_CLIP)
    restored = EngineeredScaler.from_json(scaler.to_json())
    assert np.allclose(restored.transform(stacked[:5]), scaler.transform(stacked[:5]))


def test_dataset_shapes_and_both_is_raw_then_eng(tmp_path):
    root = _make_root(tmp_path)
    scaler = EngineeredScaler.fit(root, days=(1, 2))
    datasets = {
        rep: FactorialWindowDataset(root, [7], HORIZON, rep, scaler, purge=0)
        for rep in ("raw", "eng", "both")
    }
    assert len({len(d) for d in datasets.values()}) == 1
    for rep, ds in datasets.items():
        x, y, _group, _end = ds[5]
        assert tuple(x.shape) == (100, channel_count(rep))
        assert 0 <= y <= 2
    raw_x, _, _, raw_end = datasets["raw"][5]
    eng_x, _, _, eng_end = datasets["eng"][5]
    both_x, _, _, both_end = datasets["both"][5]
    assert raw_end == eng_end == both_end
    assert torch.equal(both_x[:, :40], raw_x) and torch.equal(both_x[:, 40:], eng_x)
    with pytest.raises(ValueError, match="scaler"):
        FactorialWindowDataset(root, [7], HORIZON, "eng", None)


def test_windows_stay_inside_segments(tmp_path):
    root = _make_root(tmp_path)
    ds = FactorialWindowDataset(root, [7], HORIZON, "raw", None)
    for index in range(0, len(ds), 37):
        _, _, group, end = ds[index]
        start = int(group[1]) * SEGMENT_ROWS
        assert start + 99 <= end <= start + SEGMENT_ROWS - 2


def test_flat_matrix_alignment_lags_and_column_counts(tmp_path):
    root = _make_root(tmp_path)
    raw_x, raw_y, raw_groups = flat_matrix(root, [7], HORIZON, "raw")
    eng_x, eng_y, _ = flat_matrix(root, [7], HORIZON, "eng")
    both_x, both_y, _ = flat_matrix(root, [7], HORIZON, "both")
    assert raw_x.shape[1] == 40 * len(RAW_LAGS) == 280
    assert eng_x.shape[1] == 29 and both_x.shape[1] == 309
    assert len(raw_x) == len(eng_x) == len(both_x)
    assert np.array_equal(raw_y, eng_y) and np.array_equal(raw_y, both_y)
    ds = FactorialWindowDataset(root, [7], HORIZON, "raw", None)
    assert len(ds) == len(raw_x)
    assert [flat_feature_names(r).__len__() for r in ("raw", "eng", "both")] == [280, 29, 309]
    matrix = np.load(root / "day_07.npy")
    _, _, group, end = ds[10]
    for slot, lag in enumerate(RAW_LAGS):
        assert np.allclose(raw_x[10, slot * 40 : (slot + 1) * 40], matrix[end - lag, :40])
    assert np.array_equal(raw_groups[10], group.numpy())
    assert np.allclose(both_x[:, 280:], eng_x)


def test_deeplob_fused_uses_side_channels_and_raw_model_is_unchanged():
    assert count_parameters(DeepLOB()) == 143907
    torch.manual_seed(0)
    fused = DeepLOBFused(side_features=29).eval()
    x = torch.randn(3, 100, 69)
    base = fused(x)
    assert tuple(base.shape) == (3, 3)
    perturbed = x.clone()
    perturbed[:, -1, 40:] += 1.0
    assert not torch.allclose(base, fused(perturbed))
    # the side channels of the earliest, trunk-discarded steps must not matter
    early = x.clone()
    early[:, :10, 40:] += 5.0
    assert torch.allclose(base, fused(early), atol=1e-6)
    expected_extra = 4 * 64 * 29  # input-to-hidden weights for 29 extra LSTM inputs
    assert count_parameters(fused) - count_parameters(DeepLOB()) == expected_extra


def test_lstm_head_and_cell_builder():
    head = LSTMHead(features=29)
    assert tuple(head(torch.randn(2, 100, 29)).shape) == (2, 3)
    cells = [("M1", "raw"), ("M1", "eng"), ("M1", "both"), ("M2", "eng"), ("M2", "both")]
    for model, rep in cells:
        net = build_neural(model, rep, {"dropout": 0.1, "layers": 2})
        out = net(torch.randn(2, 100, channel_count(rep)))
        assert tuple(out.shape) == (2, 3)
