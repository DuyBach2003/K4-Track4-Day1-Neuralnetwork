"""optimizer.py — chọn bộ tối ưu, bộ lập lịch lr và cắt gradient (dùng chung cho mọi thí nghiệm).

Được dùng torch.optim.* và torch.nn.utils.clip_grad_norm_ (xem README mục 5).

Công thức (slide Chương 4):
    SGD            : w <- w - lr * g
    SGD + momentum : v <- mu * v + g ;  w <- w - lr * v          (dạng PyTorch)
    Adam           : m <- b1 m + (1-b1) g ; v <- b2 v + (1-b2) g^2 ; w <- w - lr * m_hat / (sqrt(v_hat) + eps)
    AdamW          : như Adam nhưng suy giảm trọng số tách riêng: w <- w - lr * wd * w - lr * m_hat / (sqrt(v_hat) + eps)
"""
from __future__ import annotations

import math

import torch

OPTIMIZERS = ("sgd", "sgd_momentum", "adam", "adamw")
SCHEDULERS = (None, "cosine", "warmup")


def build_optimizer(name: str, params, lr: float, weight_decay: float = 0.0,
                    momentum: float = 0.9, betas=(0.9, 0.999), eps: float = 1e-8):
    """Trả về một torch.optim.Optimizer.

    Chú ý: weight_decay của Adam là L2 trộn vào gradient (g <- g + wd·w, rồi bị chia cho √v̂),
    còn của AdamW là suy giảm tách riêng (w <- w - lr·wd·w, không đi qua √v̂).
    """
    if name not in OPTIMIZERS:
        raise ValueError(f"optimizer phải thuộc {OPTIMIZERS}, nhận {name!r}")
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd_momentum":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


def build_scheduler(optimizer, name: str | None, total_steps: int, warmup_steps: int = 0, **kwargs):
    """Bộ lập lịch lr, gọi scheduler.step() SAU MỖI BƯỚC cập nhật (không phải mỗi epoch).

    None     : lr hằng
    "cosine" : (tuỳ chọn khởi động tuyến tính warmup_steps bước) rồi giảm theo cosine từ lr về 0 ở total_steps
    "warmup" : khởi động tuyến tính từ ~0 lên lr trong warmup_steps bước, sau đó giữ hằng
               (đi kèm quy tắc tăng lr theo lô: lô ×k thì lr ×k, slide Chương 4)
    """
    if name not in SCHEDULERS:
        raise ValueError(f"scheduler phải thuộc {SCHEDULERS}, nhận {name!r}")
    if name is None:
        return None

    def warm(step):
        return min(1.0, (step + 1) / warmup_steps) if warmup_steps > 0 else 1.0

    if name == "warmup":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, warm)

    def cosine(step):
        if step < warmup_steps:
            return warm(step)
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, cosine)


def clip_gradients(params, max_norm: float | None) -> torch.Tensor:
    """Cắt gradient theo chuẩn L2 TOÀN CỤC (g <- g · min(1, c/‖g‖)) và TRẢ VỀ chuẩn TRƯỚC KHI cắt.

    Trả về tensor 0 chiều trên device (không gọi .item() ở mỗi bước để khỏi ép CPU chờ GPU);
    train.py gom lại và chỉ đọc về CPU định kỳ. Giá trị này là `grad_norm` được ghi log.
    Khi dùng FP16 + GradScaler: phải scaler.unscale_(optimizer) TRƯỚC khi gọi hàm này,
    nếu không ‖g‖ là chuẩn của gradient đã nhân hệ số s.
    """
    params = [p for p in params if p.grad is not None]
    if max_norm is None:
        # chỉ đo, không cắt
        return torch.nn.utils.get_total_norm([p.grad for p in params], norm_type=2.0)
    return torch.nn.utils.clip_grad_norm_(params, max_norm)
