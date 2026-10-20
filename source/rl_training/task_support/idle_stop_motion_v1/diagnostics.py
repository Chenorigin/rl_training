"""Select teacher diagnostics and export their scalars without per-field syncs."""

from __future__ import annotations

from collections.abc import Collection

import torch


_METADATA_MARKERS = (
    "__weighted_total__/",
    "__weighted_count__/",
    "__rms_total__/",
    "__rms_count__/",
    "__minimum_value__/",
    "__minimum_count__/",
)


def _metadata_target(key: str) -> str | None:
    """Resolve the same optional group prefix used by metric_aggregation."""
    for marker in _METADATA_MARKERS:
        if key.startswith(marker):
            return key[len(marker):]
        token = "/" + marker
        if token in key:
            prefix, suffix = key.split(token, 1)
            return prefix + "/" + suffix
    return None


def export_teacher_metrics(
    metrics: dict, core_keys: Collection[str], detailed: bool = False
) -> dict:
    """Keep requested metrics and their pooling metadata, then export scalars.

    Detailed mode keeps every field. Tensor values must contain one element;
    Python numeric scalars are accepted too. Homogeneous tensors are stacked
    before conversion, avoiding a conversion kernel for each scalar. A single
    host transfer prevents CurriculumManager synchronizing every tensor field.
    """
    selected = metrics if detailed else {
        key: value for key, value in metrics.items()
        if key in core_keys or _metadata_target(key) in core_keys
    }
    if not selected:
        return {}

    output = dict.fromkeys(selected)
    groups = {}
    for key, value in selected.items():
        if not isinstance(value, torch.Tensor):
            if isinstance(value, complex):
                raise ValueError(f"Teacher diagnostic {key!r} must be real")
            output[key] = value if isinstance(value, (bool, int, float)) else float(value)
            continue
        if value.is_complex():
            raise ValueError(f"Teacher diagnostic {key!r} must be real")
        if value.numel() != 1:
            raise ValueError(f"Teacher diagnostic {key!r} must be scalar, got shape {tuple(value.shape)}")
        groups.setdefault((value.device, value.dtype), []).append((key, value.detach().reshape(())))

    chunks, layouts = [], []
    device = None
    offset = 0
    for (group_device, _), fields in groups.items():
        keys, tensors = zip(*fields)
        values = torch.stack(tensors)
        if group_device.type == "cpu":
            output.update(zip(keys, values.tolist()))
            continue
        if device is None:
            device = group_device
        # Diagnostic integer counts stay far below float64's exact range.
        chunks.append(values.to(device=device, dtype=torch.float64))
        layouts.append((keys, offset, len(keys)))
        offset += len(keys)
    if chunks:
        packed = (chunks[0] if len(chunks) == 1 else torch.cat(chunks)).detach().cpu()
        for keys, start, size in layouts:
            output.update(zip(keys, packed[start:start + size].tolist()))
    return output
