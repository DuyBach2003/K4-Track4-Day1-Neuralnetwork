"""data.py — nạp train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.

Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).

Quy ước dữ liệu (xem README mục 2 và 3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import train_test_split

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)
N_FEATURES = 54
N_CLASSES = 7
N_TRAIN_FULL, N_EVAL = 464_809, 116_203


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz.

    Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id
    """
    tr = np.load(f"{processed_dir}/train.npz")
    ev = np.load(f"{processed_dir}/eval.npz")
    X_train_full, y_train_full = tr["X"], tr["y"]
    X_eval, y_eval, eval_row_id = ev["X"], ev["y"], ev["row_id"]

    for name, X, y, n in (("train", X_train_full, y_train_full, N_TRAIN_FULL),
                          ("eval", X_eval, y_eval, N_EVAL)):
        assert X.shape == (n, N_FEATURES) and X.dtype == np.float32, f"{name}: X sai shape/dtype {X.shape} {X.dtype}"
        assert y.shape == (n,) and y.dtype == np.int64, f"{name}: y sai shape/dtype {y.shape} {y.dtype}"
        assert y.min() == 0 and y.max() == N_CLASSES - 1, f"{name}: nhãn phải là 0..6"
    assert eval_row_id.shape == (N_EVAL,) and len(np.unique(eval_row_id)) == N_EVAL
    return X_train_full, y_train_full, X_eval, y_eval, eval_row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval), phân tầng theo nhãn.

    Trả về: X_tr, y_tr, X_val, y_val. Mọi thí nghiệm dùng cùng seed/val_fraction nên cùng một phép tách.
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_fraction, stratify=y, random_state=seed)
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    """mean và std của N_NUMERIC cột đầu, CHỈ tính trên phần train còn lại (sau khi tách val).

    Tính trên val/eval (hay toàn bộ dữ liệu) là rò rỉ thông tin: thống kê của dữ liệu dùng để đánh giá
    lọt vào bước tiền xử lý, làm điểm val/eval lạc quan hơn thực tế.
    """
    num = X_tr[:, :N_NUMERIC].astype(np.float64)  # float64 để cộng dồn 371k giá trị không mất chính xác
    mean = num.mean(axis=0)
    std = num.std(axis=0)
    std[std == 0] = 1.0  # cột hằng (không có ở dữ liệu này) thì chỉ trừ mean, tránh chia 0
    return mean.astype(np.float32), std.astype(np.float32)


def apply_standardizer(X, mean, std):
    """Bản sao của X với 10 cột đầu = (x - mean) / std; 44 cột nhị phân giữ nguyên."""
    Xs = X.copy()
    Xs[:, :N_NUMERIC] = (Xs[:, :N_NUMERIC] - mean) / std
    return Xs


def prepare_data(device: str, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed", verbose: bool = True) -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader).

    Trả về dict gồm tensor trên device: X_tr, y_tr, X_val, y_val, X_eval, y_eval (y là int64),
    và numpy: eval_row_id, mean, std, class_counts_tr.
    """
    X_full, y_full, X_eval, y_eval, eval_row_id = load_split(processed_dir)
    X_tr, y_tr, X_val, y_val = make_val_split(X_full, y_full, val_fraction, seed)
    mean, std = fit_standardizer(X_tr)  # chỉ X_tr, không X_val/X_eval
    X_tr, X_val, X_eval = (apply_standardizer(X, mean, std) for X in (X_tr, X_val, X_eval))

    def to_dev(a, dtype):
        return torch.as_tensor(a, dtype=dtype, device=device)

    data = dict(
        X_tr=to_dev(X_tr, torch.float32), y_tr=to_dev(y_tr, torch.int64),
        X_val=to_dev(X_val, torch.float32), y_val=to_dev(y_val, torch.int64),
        X_eval=to_dev(X_eval, torch.float32), y_eval=to_dev(y_eval, torch.int64),
        eval_row_id=eval_row_id, mean=mean, std=std,
        class_counts_tr=np.bincount(y_tr, minlength=N_CLASSES),
    )

    if verbose:
        print(f"train full {len(X_full):,} -> X_tr {X_tr.shape}, X_val {X_val.shape}; X_eval {X_eval.shape}")
        print("tỉ lệ lớp (%)   " + "  ".join(f"{c:>5d}" for c in range(N_CLASSES)))
        for name, y in (("train", y_tr), ("val", y_val), ("eval", y_eval)):
            p = 100 * np.bincount(y, minlength=N_CLASSES) / len(y)
            print(f"  {name:<12s} " + "  ".join(f"{v:5.2f}" for v in p))
        major = int(np.bincount(y_tr).argmax())
        print(f"'luôn đoán lớp đa số' (lớp {major}, chọn theo train) -> accuracy trên val = {np.mean(y_val == major):.4f}")
    return data


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Sinh từng cặp (xb, yb), thay cho DataLoader.

    Hoán vị được sinh trên CPU bằng `generator` (CPU) rồi mới chuyển lên device, nên thứ tự lô chỉ phụ thuộc
    seed, giống nhau trên CPU/CUDA/MPS. Lô cuối nhỏ hơn batch_size được GIỮ LẠI (không drop_last), để mỗi
    epoch dùng đúng toàn bộ mẫu train; với batch 512 lô cuối có 371 847 mod 512 = 135 mẫu.
    """
    n = len(X)
    if shuffle:
        perm = torch.randperm(n, generator=generator).to(X.device)
    else:
        perm = torch.arange(n, device=X.device)
    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        yield X[idx], y[idx]
