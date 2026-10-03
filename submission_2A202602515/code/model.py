"""model.py — MLP cho bài toán 7 lớp, shape cố định (xem README mục 3 và GUIDE, "Quy định kiến trúc"):

    x (B, 54) -> Linear(54, h1) -> ReLU -> [Dropout] -> Linear(h1, h2) -> ReLU -> [Dropout]
              -> ... -> Linear(h_last, 7) -> logits (B, 7)

Quy tắc:
  - Lớp cuối ra logit thô, KHÔNG softmax trong model (softmax nằm trong hàm mất mát).
  - Dropout chỉ đặt sau ReLU của lớp ẩn; không đặt trên đầu vào hay logit.
  - Mọi nn.Linear đều có bias. Không BatchNorm, không residual.
  - Số tham số phải khớp EXPECTED_PARAMS bên dưới.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Số tham số bắt buộc ứng với từng kiến trúc (in_features=54, num_classes=7)
EXPECTED_PARAMS = {
    (256, 128): 47_879,        # M-base  (baseline)
    (512, 256): 161_287,       # M-wide  (tuỳ chọn)
    (256, 128, 64): 55_687,    # M-deep  (tuỳ chọn)
    # Không thuộc bảng kiến trúc của README: mạng RẤT rộng, CHỈ dùng cho thí nghiệm tốc độ mixed precision
    # (GUIDE, Chủ đề 6, phần tuỳ chọn). 54·2048+2048 + 2048·2048+2048 + 2048·7+7
    (2048, 2048): 4_323_335,   # M-xwide
}

INITS = ("zeros", "normal", "xavier", "he", "default")


class MLP(nn.Module):
    """MLP theo quy định ở đầu file.

    Args:
        hidden:   tuple số nơ-ron các lớp ẩn, ví dụ (256, 128)
        dropout:  xác suất TẮT nơ-ron q (nn.Dropout dùng p chính là xác suất tắt); 0.0 = không dùng
        init:     "zeros" | "normal" | "xavier" | "he" | "default"
    """

    def __init__(self, hidden=(256, 128), dropout: float = 0.0, init: str = "he",
                 in_features: int = 54, num_classes: int = 7):
        super().__init__()
        layers: list[nn.Module] = []
        n_in = in_features
        for h in hidden:
            layers += [nn.Linear(n_in, h), nn.ReLU()]
            if dropout > 0:            # q = 0 thì không thêm lớp Dropout (tương đương, đỡ một kernel)
                layers.append(nn.Dropout(dropout))
            n_in = h
        layers.append(nn.Linear(n_in, num_classes))   # logit thô, không softmax
        self.net = nn.Sequential(*layers)
        self.hidden, self.dropout, self.init = tuple(hidden), dropout, init
        init_weights(self, init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 54) float32  ->  logits: (B, 7)."""
        return self.net(x)


def init_weights(model: nn.Module, init: str) -> None:
    """Khởi tạo W của MỌI nn.Linear; bias = 0 (trừ "default", giữ nguyên mặc định của PyTorch).

    "zeros"   : W = 0
    "normal"  : W ~ N(0, 0.01^2)
    "xavier"  : nn.init.xavier_normal_  -> Var[W] = 2/(n_in + n_out)  (công thức Glorot gốc, KHÔNG phải 1/n_in)
    "he"      : nn.init.kaiming_normal_(nonlinearity="relu") -> Var[W] = 2/n_in  (fan_in)
    "default" : giữ khởi tạo mặc định của nn.Linear (kaiming_uniform_ với a=√5 -> W ~ U(±1/√n_in),
                tức Var[W] = 1/(3·n_in), bias ~ U(±1/√n_in)); KHÔNG phải He.
    """
    if init not in INITS:
        raise ValueError(f"init phải thuộc {INITS}, nhận {init!r}")
    if init == "default":
        return
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if init == "zeros":
                nn.init.zeros_(m.weight)
            elif init == "normal":
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
            elif init == "xavier":
                nn.init.xavier_normal_(m.weight)
            elif init == "he":
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)


def count_params(model: nn.Module) -> int:
    """Tổng số tham số huấn luyện được. Dùng để assert với EXPECTED_PARAMS ngay sau khi tạo model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def activation_stats(model: nn.Module, x: torch.Tensor) -> list[float]:
    """Độ lệch chuẩn của kích hoạt theo lớp, ở chế độ eval (dropout tắt).

    Lưu std SAU MỖI ReLU (đầu ra của từng lớp ẩn) và std của logit (đầu ra Linear cuối),
    nên danh sách có len(hidden) + 1 phần tử.
    """
    was_training = model.training
    model.eval()
    stds, h = [], x
    layers = list(model.net)
    for i, layer in enumerate(layers):
        h = layer(h)
        if isinstance(layer, nn.ReLU) or i == len(layers) - 1:
            stds.append(h.float().std().item())
    model.train(was_training)
    return stds
