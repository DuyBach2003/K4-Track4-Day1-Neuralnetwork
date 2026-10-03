"""train.py — đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.

Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).
Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import contextlib
import copy
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, build_scheduler, clip_gradients

N_CLASSES = 7

# Cấu hình mặc định = BASELINE (M-base). `lr` được chọn bằng val trong notebook (Part 2) rồi điền vào.
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=None,                   # chọn bằng val trong notebook, không dùng eval
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    scheduler=None,            # None | "cosine" | "warmup"  (xem optimizer.build_scheduler)
    warmup_steps=0,
    seed=1,
    notes="",
)

CHECK_EVERY = 50   # cứ 50 bước đọc loss về CPU một lần để phát hiện NaN/inf (đọc mỗi bước sẽ ép CPU chờ GPU)
GN_STRIDE = 10     # lưu chuẩn gradient của 1/10 số bước vào history (để vẽ "gai"), trung bình thì tính trên mọi bước


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (torch.manual_seed cũng đặt cho CUDA và MPS)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------------------------- thiết bị
def _sync(device: torch.device) -> None:
    """Chờ GPU chạy xong mọi kernel đã xếp hàng, để đo thời gian đúng."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _mem_allocated(device: torch.device) -> int | None:
    if device.type == "cuda":
        return torch.cuda.memory_allocated()
    if device.type == "mps":
        return torch.mps.current_allocated_memory()
    return None


def _autocast(device: torch.device, precision: str):
    """Ngữ cảnh autocast cho forward + loss. Tham số vẫn FP32; chỉ phép toán được hạ độ chính xác."""
    if precision == "fp32":
        return contextlib.nullcontext()
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[precision]
    return torch.autocast(device_type=device.type, dtype=dtype)


# ----------------------------------------------------------------------------------------------- chỉ số
def confusion_matrix_t(y_true: torch.Tensor, y_pred: torch.Tensor, k: int = N_CLASSES) -> torch.Tensor:
    """Ma trận nhầm lẫn k×k (hàng = nhãn thật, cột = dự đoán), tính ngay trên device bằng bincount."""
    return torch.bincount(y_true * k + y_pred, minlength=k * k).reshape(k, k)


def per_class_scores(cm: np.ndarray):
    """precision, recall, F1 từng lớp — đúng công thức của scripts/evaluate.py (bằng 0 nếu mẫu số bằng 0)."""
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return prec, rec, f1


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0.

    cm: ma trận nhầm lẫn (7, 7), hàng = nhãn thật, cột = dự đoán.
    """
    return float(per_class_scores(cm)[2].mean())


def compute_loss(logits, y, loss_name: str):
    """Loss trung bình của một lô.

    "ce"  : F.cross_entropy(logits, y) — nhận logit thô (log-softmax nằm bên trong) và nhãn int64.
    "mse" : F.mse_loss(logits, onehot(y)) với reduction="mean", tức trung bình trên MỌI B×7 phần tử,
            không có hệ số 1/2 (giống nn.MSELoss). Gradient theo mỗi logit: 2(z − y)/(7B).
    Logit được ép về FP32 trước khi tính loss để exp/log không tràn số khi chạy autocast FP16/BF16.
    """
    logits = logits.float()
    if loss_name == "ce":
        return F.cross_entropy(logits, y)
    if loss_name == "mse":
        return F.mse_loss(logits, F.one_hot(y, N_CLASSES).float())
    raise ValueError(f"loss phải là 'ce' hoặc 'mse', nhận {loss_name!r}")


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    """Nhãn dự đoán int64 (N,) = argmax của logits, ở chế độ eval (dropout tắt), FP32."""
    model.eval()
    return torch.cat([model(X[i:i + batch_size]).argmax(dim=1) for i in range(0, len(X), batch_size)])


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """dict(loss, acc, macro_f1) ở chế độ eval() và no_grad, FP32.

    loss = trung bình trên toàn bộ N mẫu (cộng loss_lô × kích thước lô rồi chia N; với MSE đây đúng bằng
    trung bình trên N×7 phần tử, cùng định nghĩa với loss lúc huấn luyện).
    """
    model.eval()
    sums, cm = [], torch.zeros(N_CLASSES, N_CLASSES, dtype=torch.int64, device=X.device)
    for i in range(0, len(X), batch_size):
        xb, yb = X[i:i + batch_size], y[i:i + batch_size]
        logits = model(xb)
        sums.append(compute_loss(logits, yb, loss_name) * len(xb))
        cm += confusion_matrix_t(yb, logits.argmax(dim=1))
    loss = torch.stack(sums).cpu().double().sum().item() / len(X)   # MPS không có float64: về CPU trước
    cm = cm.cpu().numpy()
    return dict(loss=loss, acc=float(np.trace(cm) / cm.sum()), macro_f1=macro_f1_from_confusion(cm))


# ----------------------------------------------------------------------------------------------- huấn luyện
def build_model(cfg: dict, device) -> MLP:
    """Tạo MLP theo cfg và kiểm tra số tham số đúng quy định."""
    hidden = tuple(cfg["hidden"])
    model = MLP(hidden=hidden, dropout=cfg["dropout"], init=cfg["init"])
    n = count_params(model)
    assert n == EXPECTED_PARAMS[hidden], f"{hidden}: {n} tham số, quy định {EXPECTED_PARAMS[hidden]}"
    return model.to(device)


def run_experiment(cfg: dict, data: dict, verbose: bool = True) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt.

    Args:
        cfg : dict cấu hình (khoá như DEFAULT_CFG; khoá thiếu lấy từ DEFAULT_CFG)
        data: kết quả của data.prepare_data (tensor X_tr, y_tr, X_val, y_val trên device)

    Trả về {"cfg", "history", "summary", "best_state"}; khoá của summary trùng tên cột experiments.xlsx
    (cộng vài khoá phụ). best_state = state_dict (trên CPU) ở epoch có val_loss thấp nhất.
    TUYỆT ĐỐI không dùng X_eval ở đây: epoch tốt nhất và mọi lựa chọn chỉ dựa trên val.
    """
    cfg = {**DEFAULT_CFG, **cfg}
    assert cfg["lr"] is not None, "chưa đặt lr"
    X_tr, y_tr, X_val, y_val = data["X_tr"], data["y_tr"], data["X_val"], data["y_val"]
    device = X_tr.device
    loss_name, clip, precision = cfg["loss"], cfg["clip_norm"], cfg["precision"]

    # ---- 0. seed, model, optimizer
    set_seed(cfg["seed"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    mem0 = _mem_allocated(device)          # bộ nhớ trước khi tạo model (đã gồm toàn bộ dữ liệu trên device)
    model = build_model(cfg, device)
    optimizer = build_optimizer(cfg["optimizer"], model.parameters(), lr=cfg["lr"],
                                weight_decay=cfg["weight_decay"], momentum=cfg["momentum"])
    steps_per_epoch = math.ceil(len(X_tr) / cfg["batch"])
    scheduler = build_scheduler(optimizer, cfg["scheduler"], total_steps=steps_per_epoch * cfg["epochs"],
                                warmup_steps=cfg["warmup_steps"])
    scaler = torch.amp.GradScaler(device.type) if precision == "fp16" else None
    gen = torch.Generator().manual_seed(cfg["seed"])   # điều khiển thứ tự lô (xem iterate_batches)

    # ---- 1. loss bước 0 trên val, TRƯỚC bước cập nhật đầu tiên (kỳ vọng ≈ ln 7 ≈ 1.946 với CE)
    step0 = evaluate(model, X_val, y_val, loss_name)

    keys = ("epoch", "train_loss", "train_acc", "train_loss_running", "val_loss", "val_acc", "val_macro_f1",
            "grad_norm", "grad_norm_max", "frac_clipped", "lr", "epoch_time_s")
    hist = {k: [] for k in keys}
    hist["grad_norm_steps"], hist["grad_norm_steps_stride"] = [], GN_STRIDE
    best = dict(val_loss=math.inf, epoch=0, state=None)
    diverged, peak_sampled, n_skipped = False, 0, 0
    t_start = time.perf_counter()

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        _sync(device)
        t0 = time.perf_counter()
        gns, losses = [], []
        for step, (xb, yb) in enumerate(iterate_batches(X_tr, y_tr, cfg["batch"], gen)):
            with _autocast(device, precision):            # chỉ bọc forward + loss
                logits = model(xb)
                loss = compute_loss(logits, yb, loss_name)
            if epoch == 1 and step < 5 and mem0 is not None and device.type == "mps":
                # MPS không có max_memory_allocated: lấy mẫu sau forward (activation + grad cũ + trạng thái optimizer)
                peak_sampled = max(peak_sampled, _mem_allocated(device))
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)   # LUÔN unscale trước khi đo/cắt, để grad_norm là chuẩn thật
            else:
                loss.backward()
            gn = clip_gradients(model.parameters(), clip)    # chuẩn TRƯỚC khi cắt
            if scaler is not None:
                scaler.step(optimizer)       # bỏ qua bước nếu gradient có inf/NaN (tràn FP16)
                scaler.update()
            else:
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            gns.append(gn.detach())
            losses.append(loss.detach())
            if (step + 1) % CHECK_EVERY == 0 and not torch.isfinite(torch.stack(losses[-CHECK_EVERY:])).all().item():
                diverged = True
                break
        if not diverged and not torch.isfinite(torch.stack(losses)).all().item():
            diverged = True
        _sync(device)
        epoch_time = time.perf_counter() - t0               # chỉ thời gian huấn luyện, không gồm đánh giá

        gn_t = torch.stack(gns).float().cpu()
        finite = torch.isfinite(gn_t)
        n_skipped += int((~finite).sum())                  # bước có gradient inf/NaN (FP16 tràn số)
        gn_f = gn_t[finite]
        tr = evaluate(model, X_tr, y_tr, loss_name)         # train loss đo ở eval mode, cùng thang với val
        va = evaluate(model, X_val, y_val, loss_name)
        if not math.isfinite(va["loss"]):
            diverged = True

        hist["epoch"].append(epoch)
        hist["train_loss"].append(tr["loss"])
        hist["train_acc"].append(tr["acc"])
        hist["train_loss_running"].append(torch.stack(losses).float().mean().item())  # loss lúc train (có dropout)
        hist["val_loss"].append(va["loss"])
        hist["val_acc"].append(va["acc"])
        hist["val_macro_f1"].append(va["macro_f1"])
        hist["grad_norm"].append(gn_f.mean().item() if len(gn_f) else float("nan"))
        hist["grad_norm_max"].append(gn_f.max().item() if len(gn_f) else float("nan"))
        hist["frac_clipped"].append(float((gn_f > clip).float().mean()) if clip is not None and len(gn_f) else 0.0)
        hist["lr"].append(optimizer.param_groups[0]["lr"])
        hist["epoch_time_s"].append(epoch_time)
        hist["grad_norm_steps"] += gn_t[::GN_STRIDE].tolist()

        if math.isfinite(va["loss"]) and va["loss"] < best["val_loss"]:
            best.update(val_loss=va["loss"], epoch=epoch,
                        state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        if verbose:
            print(f"  [{cfg['exp_id']}] ep {epoch:2d}  train {tr['loss']:.4f}  val {va['loss']:.4f}  "
                  f"acc {va['acc']:.4f}  F1 {va['macro_f1']:.4f}  |g| {hist['grad_norm'][-1]:.3f}  "
                  f"{epoch_time:.1f}s" + ("  DIVERGED" if diverged else ""))
        if diverged:
            break

    # ---- 3. tóm tắt tại best_epoch
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated()
    elif device.type == "mps":
        peak = peak_sampled
    else:
        peak = None
    b = best["epoch"] - 1   # chỉ số trong history; -1 nghĩa là không có epoch hữu hạn nào
    # epoch 1 gồm cả chi phí khởi động (biên dịch kernel Metal/CUDA lần đầu) nên không tính vào trung bình
    times = hist["epoch_time_s"][1:] or hist["epoch_time_s"]
    summary = dict(
        step0_loss=step0["loss"],
        best_val_loss=best["val_loss"] if b >= 0 else float("nan"),
        best_epoch=best["epoch"] if b >= 0 else None,
        final_train_loss=hist["train_loss"][-1],
        final_val_loss=hist["val_loss"][-1],
        val_acc=hist["val_acc"][b] if b >= 0 else step0["acc"],
        val_macro_f1=hist["val_macro_f1"][b] if b >= 0 else step0["macro_f1"],
        time_per_epoch_s=float(np.mean(times)),
        peak_mem_MB=peak / 2**20 if peak is not None else None,
        diverged=diverged,
        # khoá phụ (lưu trong JSON, không có cột riêng trong bảng)
        step0_val_acc=step0["acc"],
        train_loss_at_best=hist["train_loss"][b] if b >= 0 else float("nan"),
        final_val_acc=hist["val_acc"][-1], final_val_macro_f1=hist["val_macro_f1"][-1],
        peak_mem_train_MB=(peak - mem0) / 2**20 if peak is not None else None,
        time_per_epoch_median_s=float(np.median(times)), epoch1_time_s=hist["epoch_time_s"][0],
        n_params=count_params(model), steps_per_epoch=steps_per_epoch,
        epochs_run=len(hist["epoch"]), n_nonfinite_grad_steps=n_skipped,
        mean_frac_clipped=float(np.mean(hist["frac_clipped"])),
        total_time_s=time.perf_counter() - t_start, device=str(device), torch=torch.__version__,
    )
    if verbose:
        s = summary
        print(f"  => {cfg['exp_id']}: step0 {s['step0_loss']:.4f} | best ep {s['best_epoch']} "
              f"val_loss {s['best_val_loss']:.4f} acc {s['val_acc']:.4f} F1 {s['val_macro_f1']:.4f} | "
              f"{s['time_per_epoch_s']:.2f}s/ep" + (" | DIVERGED" if diverged else ""))
    return dict(cfg=cfg, history=hist, summary=summary, best_state=best["state"])


def overfit_tiny(data: dict, n: int = 20, steps: int = 500, lr: float = 1e-2, seed: int = 0) -> list[float]:
    """Phép thử sức khoẻ: quá khớp n mẫu train (dropout tắt, Adam). Trả về loss từng bước (phải về ≈ 0)."""
    set_seed(seed)
    device = data["X_tr"].device
    idx = torch.randperm(len(data["X_tr"]), generator=torch.Generator().manual_seed(seed))[:n].to(device)
    xb, yb = data["X_tr"][idx], data["y_tr"][idx]
    model = MLP(hidden=(256, 128), dropout=0.0, init="he").to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    curve = []
    model.train()
    for _ in range(steps):
        loss = compute_loss(model(xb), yb, "ce")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        curve.append(loss.detach())
    acc = (predict(model, xb) == yb).float().mean().item()
    curve = torch.stack(curve).cpu().tolist()
    print(f"quá khớp {n} mẫu: loss {curve[0]:.4f} -> {curve[-1]:.6f} sau {steps} bước; accuracy trên {n} mẫu = {acc:.2f}")
    return curve


# ----------------------------------------------------------------------------------------------- nộp bài
def write_predictions(row_id, preds, path: str) -> None:
    """CSV có tiêu đề `row_id,pred` cho scripts/evaluate.py; đủ mọi dòng eval, mỗi row_id đúng một lần."""
    row_id, preds = np.asarray(row_id), np.asarray(preds)
    assert len(row_id) == len(preds) and len(np.unique(row_id)) == len(row_id)
    assert preds.min() >= 0 and preds.max() <= N_CLASSES - 1
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"row_id": row_id.astype(np.int64), "pred": preds.astype(np.int64)}).to_csv(path, index=False)


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str,
               repo_root: str | None = None, out_json: str | None = None) -> dict | None:
    """Dùng cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán TOÀN BỘ eval, ghi predictions,
    rồi (nếu có repo_root) chạy đúng `scripts/evaluate.py` và trả về nội dung JSON nó ghi ra."""
    cfg = {**DEFAULT_CFG, **cfg}
    model = build_model(cfg, data["X_eval"].device)
    model.load_state_dict(result["best_state"])
    preds = predict(model, data["X_eval"])              # fp32, eval mode
    write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
    print(f"đã ghi {pred_path} ({len(preds):,} dòng) từ {cfg['exp_id']}, epoch {result['summary']['best_epoch']}")
    if repo_root is None:
        return None
    out_json = out_json or str(Path(pred_path).with_suffix(".json"))
    cmd = [sys.executable, "scripts/evaluate.py", "--pred", str(Path(pred_path).resolve()),
           "--out", str(Path(out_json).resolve())]
    proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout)
    with open(out_json) as f:
        return json.load(f)


def strip_state(result: dict) -> dict:
    """Bản sao của result không có best_state (để lưu/so sánh nhẹ)."""
    return {k: copy.deepcopy(v) for k, v in result.items() if k != "best_state"}
