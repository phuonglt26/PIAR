import math
from typing import List, Optional
import numpy as np
import torch


def grad_influence_stats(
    scalar_loss: torch.Tensor,
    model: torch.nn.Module,
    eps: float = 1e-10,
    group_by: Optional[List[str]] = None,  # ví dụ: ["encoder.", "decoder.", "lm_head"]
    retain_graph: bool = True,
):
    """
    Trả về:
      - ratio_used: % tham số có grad != None
      - ratio_nonzero: % tham số có ||grad|| > eps
      - grad_norm_{mean,median,max}
      - nếu group_by: thống kê theo nhóm prefix
    """
    if scalar_loss is None:
        return {"ratio_used": 0.0, "ratio_nonzero": 0.0}

    base = getattr(model, "module", model)

    named = [(n, p) for n, p in base.named_parameters() if p.requires_grad]
    if not named:
        return {"ratio_used": 0.0, "ratio_nonzero": 0.0}

    params = [p for _, p in named]
    grads = torch.autograd.grad(
        scalar_loss,
        params,
        retain_graph=retain_graph,
        allow_unused=True,
        create_graph=False,
    )

    @torch.no_grad()
    def _safe_norm(x: torch.Tensor) -> float:
        try:
            v = x.norm().item()
            if not math.isfinite(v):
                return float("nan")
            return float(v)
        except Exception:
            return float("nan")

    used = 0
    nonzero = 0
    norms = []
    by_group = {g: [] for g in (group_by or [])}

    for (name, _), g in zip(named, grads):
        if g is not None:
            used += 1
            n = _safe_norm(g)
            norms.append(n)
            if n > eps:
                nonzero += 1
            if group_by:
                for gname in group_by:
                    if gname in name:
                        by_group[gname].append(n)

    total = len(named)
    out = {
        "ratio_used": used / total,
        "ratio_nonzero": nonzero / total,
        "grad_norm_mean": float(np.nanmean(norms)) if norms else 0.0,
        "grad_norm_median": float(np.nanmedian(norms)) if norms else 0.0,
        "grad_norm_max": float(np.nanmax(norms)) if norms else 0.0,
        "total_params": total,
        "used_params": used,
        "nonzero_params": nonzero,
    }
    if group_by:
        for gname, arr in by_group.items():
            if arr:
                out[f"{gname}::mean"] = float(np.nanmean(arr))
                out[f"{gname}::median"] = float(np.nanmedian(arr))
                out[f"{gname}::max"] = float(np.nanmax(arr))
                out[f"{gname}::pct_nonzero"] = float(
                    sum(1 for v in arr if v > eps) / max(1, len(arr))
                )
            else:
                out[f"{gname}::mean"] = 0.0
                out[f"{gname}::median"] = 0.0
                out[f"{gname}::max"] = 0.0
                out[f"{gname}::pct_nonzero"] = 0.0
    return out


def grad_influence_ratio(scalar_loss: torch.Tensor, model: torch.nn.Module) -> float:
    """
    Trả về tỉ lệ tham số có grad khác None khi phân biệt theo scalar_loss.
    Dùng autograd.grad để KHÔNG làm bẩn .grad của mô hình.
    """
    if scalar_loss is None:
        return 0.0

    base = getattr(model, "module", model)
    params = [p for p in base.parameters() if p.requires_grad]

    if not params:
        return 0.0

    grads = torch.autograd.grad(
        scalar_loss, params, retain_graph=True, allow_unused=True, create_graph=False
    )

    @torch.no_grad()
    def _count_non_none(grads):
        cnt = 0
        for g in grads:
            if g is not None and torch.isfinite(g).any():
                cnt += 1
        return cnt

    nn = _count_non_none(grads)
    return float(nn) / float(len(params))


def print_progress(current, total, bar_length=40):
    """
    Hiển thị progress bar đẹp mắt trong terminal.

    Args:
        current (int): số lượng đã xử lý (bắt đầu từ 1).
        total (int): tổng số lượng cần xử lý.
        bar_length (int): độ dài của thanh progress bar.
    """
    progress = current / total
    filled_length = int(bar_length * progress)
    bar = "█" * filled_length + "-" * (bar_length - filled_length)
    print(f"\rProcessing: |{bar}| {progress*100:.2f}% " f"({current}/{total})", end="")
    if current == total:
        print("\nDone!")


# import time


# def print_progress(current, total, start_time, bar_length=40):
#     progress = current / total
#     filled_length = int(bar_length * progress)
#     bar = "█" * filled_length + "-" * (bar_length - filled_length)

#     elapsed = time.time() - start_time
#     avg_time = elapsed / current if current > 0 else 0
#     eta = avg_time * (total - current)
#     h = eta // 3600
#     m = (eta % 3600) // 60
#     s = eta % 60

#     print(
#         f"\rProcessing: |{bar}| {progress*100:.2f}% "
#         f"({current}/{total}) | elapsed: {elapsed:.1f} | ETA: {f"{int(h)}:{int(m)}:{s:02d}"}",
#         end="",
#     )

#     if current == total:
#         print("\nDone!")
