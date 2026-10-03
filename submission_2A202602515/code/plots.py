"""plots.py — ảnh cho từng thí nghiệm (figures/<exp_id>.png) và ảnh chồng theo nhóm (figures/compare_<nhóm>.png).

Khi notebook chạy trong code/, lưu vào "../figures/".
Màu: bảng màu phân loại cố định (gán theo thứ tự, không xoay vòng); chữ luôn dùng màu mực, không dùng màu series.
"""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

STYLE = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "legend.frameon": False, "legend.labelcolor": INK2, "font.size": 9.5, "axes.titlesize": 10.5,
    "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
}

LABELS = {
    "train_loss": "train loss (eval mode)", "val_loss": "val loss", "val_acc": "val accuracy",
    "val_macro_f1": "val macro-F1", "grad_norm": "‖g‖ trung bình mỗi epoch (trước clip)",
    "grad_norm_max": "‖g‖ lớn nhất mỗi epoch", "epoch_time_s": "thời gian / epoch (s)", "train_acc": "train accuracy",
}

OPT_NAMES = {"sgd": "SGD", "sgd_momentum": "SGD+mom", "adam": "Adam", "adamw": "AdamW"}


def cfg_label(cfg: dict) -> str:
    """Một dòng mô tả cấu hình chính, dùng cho tiêu đề ảnh."""
    parts = [f"{cfg['loss'].upper()}", f"{OPT_NAMES.get(cfg['optimizer'], cfg['optimizer'])} lr={cfg['lr']:g}"]
    if cfg.get("weight_decay"):
        parts.append(f"wd={cfg['weight_decay']:g}")
    parts += [f"batch={cfg['batch']}", f"{cfg['epochs']} ep", "-".join(map(str, cfg["hidden"])),
              f"init={cfg['init']}", f"dropout={cfg['dropout']:g}",
              f"clip={'none' if cfg['clip_norm'] is None else format(cfg['clip_norm'], 'g')}", cfg["precision"]]
    if cfg.get("scheduler"):
        parts.append(f"sched={cfg['scheduler']}")
    parts.append(f"seed={cfg['seed']}")
    return " · ".join(parts)


def _finite_max(values) -> float:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    return max(vals) if vals else float("nan")


def _mark_best(ax, best_epoch):
    if best_epoch:
        ax.axvline(best_epoch, color=MUTED, linewidth=1, zorder=0)


def plot_run(result: dict, path: str) -> None:
    """Một thí nghiệm -> một PNG với 3 ô:
         (1) train loss (đo ở eval mode) và val loss theo epoch
         (2) val accuracy và val macro-F1 theo epoch
         (3) chuẩn gradient (trước khi clip): từng bước (lấy mẫu 1/10, nét mảnh) + trung bình mỗi epoch
    Đường dọc xám = best_epoch (val loss thấp nhất).
    """
    cfg, h, s = result["cfg"], result["history"], result["summary"]
    ep = np.array(h["epoch"])
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

        ax = axes[0]
        ax.plot(ep, h["train_loss"], color=SERIES[0], marker="o", markersize=3.5, label="train (eval mode)")
        ax.plot(ep, h["val_loss"], color=SERIES[1], marker="o", markersize=3.5, label="val")
        ax.annotate(f"loss bước 0 (val): {s['step0_loss']:.3f}   ln 7 = 1.946" if cfg["loss"] == "ce"
                    else f"loss bước 0 (val): {s['step0_loss']:.4f}",
                    xy=(0.02, 0.03), xycoords="axes fraction", color=INK2, fontsize=8.5)
        _mark_best(ax, s["best_epoch"])
        ax.set(title=f"Loss ({cfg['loss'].upper()})", xlabel="epoch", ylabel="loss")
        ax.legend(loc="upper right")

        ax = axes[1]
        ax.plot(ep, h["val_acc"], color=SERIES[0], marker="o", markersize=3.5, label="val accuracy")
        ax.plot(ep, h["val_macro_f1"], color=SERIES[1], marker="o", markersize=3.5, label="val macro-F1")
        _mark_best(ax, s["best_epoch"])
        if s["best_epoch"]:
            ax.annotate(f"best ep {s['best_epoch']}: acc {s['val_acc']:.4f}, F1 {s['val_macro_f1']:.4f}",
                        xy=(0.02, 0.03), xycoords="axes fraction", color=INK2, fontsize=8.5)
        ax.set(title="Val accuracy / macro-F1", xlabel="epoch", ylabel="giá trị (0–1)")
        ax.legend(loc="lower right")

        ax = axes[2]
        steps = np.array(h.get("grad_norm_steps", []), dtype=float)
        if len(steps):
            spe = s["steps_per_epoch"]
            x = (np.arange(len(steps)) * h.get("grad_norm_steps_stride", 1)) / spe   # epoch thực (0..E)
            ax.plot(x, steps, color=SERIES[0], linewidth=0.6, alpha=0.35, label="từng bước (1/10 số bước)")
        ax.plot(ep - 0.5, h["grad_norm"], color=SERIES[0], marker="o", markersize=3.5, label="trung bình epoch")
        if cfg["clip_norm"] is not None:
            ax.axhline(cfg["clip_norm"], color=SERIES[1], linewidth=1.5,
                       label=f"ngưỡng clip c={cfg['clip_norm']:g} (cắt {100 * s['mean_frac_clipped']:.0f}% số bước)")
        finite = steps[np.isfinite(steps)] if len(steps) else np.array(h["grad_norm"])
        finite = finite[finite > 0]
        if len(finite) and finite.max() / max(finite.min(), 1e-12) > 50:
            ax.set_yscale("log")
        ax.set(title="Chuẩn gradient ‖g‖₂ toàn cục (trước clip)", xlabel="epoch", ylabel="‖g‖")
        ax.legend(loc="upper right", fontsize=8)

        for ax in axes:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        title = f"{cfg['exp_id']}  —  {cfg_label(cfg)}"
        if s["diverged"]:
            title += "   [DIVERGED: loss NaN/inf, dừng sớm]"
        fig.suptitle(title, color=INK, fontsize=11, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)


def plot_compare(results: list[dict], metric, path: str, title: str = "", labels: list[str] | None = None) -> None:
    """Vẽ chồng một hay nhiều chỉ số của nhiều thí nghiệm (mỗi thí nghiệm một đường, chú thích bằng exp_id).

    metric: tên một chỉ số trong history (ví dụ "val_loss") hoặc danh sách tên -> mỗi chỉ số một ô.
    Màu theo thứ tự thí nghiệm trong `results` (tối đa 8); ô "grad_norm" tự chuyển sang thang log nếu cần.
    """
    metrics = [metric] if isinstance(metric, str) else list(metric)
    assert len(results) <= len(SERIES), "tối đa 8 đường mỗi ảnh; tách thành nhiều ảnh nếu nhiều hơn"
    labels = labels or [r["cfg"]["exp_id"] for r in results]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 4.2), squeeze=False)
        for ax, m in zip(axes[0], metrics):
            vmax = []
            for i, (r, lab) in enumerate(zip(results, labels)):
                h = r["history"]
                ax.plot(h["epoch"], h[m], color=SERIES[i], marker="o", markersize=3, label=lab)
                vmax.append(_finite_max(h[m]))
                if r["summary"]["diverged"]:
                    ax.plot(h["epoch"][-1], 0, marker="x", color=SERIES[i], markersize=8, clip_on=False)
            if m.startswith("grad_norm"):
                vals = [v for r in results for v in r["history"][m] if v and math.isfinite(v) and v > 0]
                if vals and max(vals) / min(vals) > 50:
                    ax.set_yscale("log")
            ax.set(title=LABELS.get(m, m), xlabel="epoch", ylabel=LABELS.get(m, m).split(" (")[0])
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        axes[0][0].legend(loc="best", fontsize=8)
        if title:
            fig.suptitle(title, color=INK, fontsize=11, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.94 if title else 1))
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)


def plot_lr_sensitivity(points: dict[str, list[tuple[float, float]]], path: str, title: str = "",
                        ylabel: str = "val macro-F1 (best epoch)", ref: tuple[float, float] | None = None) -> None:
    """Độ nhạy với lr: mỗi bộ tối ưu một đường (x = lr, thang log; y = chỉ số ở best epoch).

    ref = (mean, 2σ) của baseline -> vẽ dải nhiễu seed quanh mean để thấy chênh lệch nào vượt nhiễu.
    """
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        if ref is not None:
            m, band = ref
            ax.axhspan(m - band, m + band, color=GRID, alpha=0.6, zorder=0, label="baseline ± 2σ (seed)")
        for i, (name, pts) in enumerate(points.items()):
            pts = sorted(pts)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=SERIES[i], marker="o", markersize=5, label=name)
        ax.set_xscale("log")
        ax.set(xlabel="learning rate (log)", ylabel=ylabel, title=title)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)


def plot_activation_depth(stds_by_init: dict[str, list[float]], path: str, title: str = "") -> None:
    """Độ lệch chuẩn kích hoạt theo độ sâu (một mạng ReLU sâu, bước 0) cho từng cách khởi tạo — trục y log."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        for i, (name, stds) in enumerate(stds_by_init.items()):
            stds = np.maximum(np.array(stds, dtype=float), 1e-30)   # std = 0 (zeros) vẽ ở sàn 1e-30
            ax.plot(np.arange(1, len(stds) + 1), stds, color=SERIES[i], marker="o", markersize=3, label=name)
        ax.set_yscale("log")
        ax.set(xlabel="lớp (sau ReLU; lớp cuối = logit)", ylabel="std kích hoạt (log)", title=title)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)


def show(path: str):
    """Hiển thị PNG đã lưu ngay trong notebook (an toàn cả khi chạy bằng nbconvert)."""
    from IPython.display import Image, display
    display(Image(filename=path))
