"""results_table.py — lưu kết quả từng lần chạy ra JSON, rồi điền vào experiments.xlsx từ mẫu
templates/experiment_table_template.xlsx (không gõ tay).

Tên cột của sheet "Experiments" (giữ nguyên, đúng thứ tự mẫu):
    exp_id, group, description, loss, optimizer, lr, weight_decay, batch, epochs, hidden, dropout,
    clip_norm, precision, init, seed, step0_loss, best_val_loss, best_epoch, final_train_loss,
    final_val_loss, val_acc, val_macro_f1, time_per_epoch_s, peak_mem_MB, diverged,
    eval_acc, eval_macro_f1, figure_file, notes
(các cột công thức ở cuối bảng mẫu tự tính, không ghi đè)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import openpyxl

FORMULA_COLS = {"step0_gap_vs_lnC", "gap_val_minus_train", "delta_val_f1_vs_base", "beyond_noise"}
LOSS_NAMES = {"ce": "CE", "mse": "MSE"}
OPT_NAMES = {"sgd": "SGD", "sgd_momentum": "SGD+momentum", "adam": "Adam", "adamw": "AdamW"}
TEMPLATE_ROWS = 60   # mẫu có công thức cho dòng 2..61


def _clean(x):
    """Đổi sang kiểu JSON hợp lệ: tuple -> list, NaN/inf -> None."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def save_result(result: dict, results_dir: str = "../results") -> str:
    """Ghi cfg, history, summary (KHÔNG ghi best_state) ra <results_dir>/<exp_id>.json; trả về đường dẫn."""
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    path = Path(results_dir) / f"{result['cfg']['exp_id']}.json"
    payload = {k: _clean(result[k]) for k in ("cfg", "history", "summary")}
    with open(path, "w") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)
    return str(path)


def load_results(results_dir: str = "../results") -> list[dict]:
    """Đọc mọi <exp_id>.json trong results_dir (bỏ qua file khác), sắp theo exp_id.

    None trong history/summary được đổi lại thành NaN để vẽ và tính toán như kết quả gốc.
    """
    out = []
    for p in sorted(Path(results_dir).glob("*.json")):
        with open(p) as f:
            r = json.load(f)
        if not {"cfg", "history", "summary"} <= r.keys():
            continue
        r["cfg"]["hidden"] = tuple(r["cfg"]["hidden"])
        for k, v in r["history"].items():
            if isinstance(v, list):
                r["history"][k] = [float("nan") if x is None else x for x in v]
        out.append(r)
    return sorted(out, key=lambda r: r["cfg"]["exp_id"])


def _num(x, nd=6):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return None
    return round(float(x), nd)


def to_row(result: dict, eval_scores: dict | None = None, notes: str = "") -> dict:
    """Một kết quả -> một dòng bảng (khoá trùng tên cột). Chỉ truyền eval_scores cho baseline và cấu hình cuối."""
    c, s = result["cfg"], result["summary"]
    note_parts = [p for p in (c.get("notes", ""), notes) if p]
    if c.get("scheduler"):
        note_parts.append(f"scheduler={c['scheduler']}" + (f", warmup {c['warmup_steps']} bước" if c.get("warmup_steps") else ""))
    if s.get("diverged"):
        note_parts.append(f"dừng sớm ở epoch {s['epochs_run']} vì loss NaN/inf")
    if c["optimizer"] == "sgd_momentum" and c.get("momentum", 0.9) != 0.9:
        note_parts.append(f"momentum={c['momentum']}")
    row = dict(
        exp_id=c["exp_id"], group=c["group"], description=c["description"],
        loss=LOSS_NAMES[c["loss"]], optimizer=OPT_NAMES[c["optimizer"]], lr=c["lr"],
        weight_decay=c["weight_decay"], batch=c["batch"], epochs=c["epochs"],
        hidden="-".join(str(h) for h in c["hidden"]), dropout=c["dropout"],
        clip_norm="none" if c["clip_norm"] is None else c["clip_norm"],
        precision=c["precision"], init=c["init"], seed=c["seed"],
        step0_loss=_num(s["step0_loss"]), best_val_loss=_num(s["best_val_loss"]), best_epoch=s["best_epoch"],
        final_train_loss=_num(s["final_train_loss"]), final_val_loss=_num(s["final_val_loss"]),
        val_acc=_num(s["val_acc"]), val_macro_f1=_num(s["val_macro_f1"]),
        time_per_epoch_s=_num(s["time_per_epoch_s"], 3), peak_mem_MB=_num(s["peak_mem_MB"], 1),
        diverged="Y" if s["diverged"] else "N",
        eval_acc=_num(eval_scores["accuracy"]) if eval_scores else None,
        eval_macro_f1=_num(eval_scores["macro_f1"]) if eval_scores else None,
        figure_file=f"figures/{c['exp_id']}.png", notes="; ".join(note_parts),
    )
    return row


def write_xlsx(rows: list[dict], template_path: str, out_path: str,
               seeds: list[str] | None = None, summary_notes: dict[str, str] | None = None) -> None:
    """Điền các dòng vào sheet "Experiments" của mẫu (từ dòng 2), sheet "Seeds" (exp_id baseline khác seed)
    và cột nhận xét của sheet "Summary", rồi lưu thành out_path. Giữ nguyên mọi công thức của mẫu."""
    assert len(rows) <= TEMPLATE_ROWS, f"mẫu chỉ có công thức cho {TEMPLATE_ROWS} dòng"
    assert len({r['exp_id'] for r in rows}) == len(rows), "exp_id phải duy nhất"
    wb = openpyxl.load_workbook(template_path)          # KHÔNG data_only=True (sẽ mất công thức)
    ws = wb["Experiments"]
    header = {cell.value: cell.column for cell in ws[1] if cell.value}

    # xoá giá trị mẫu (dòng baseline điền sẵn) ở các cột nhập, giữ định dạng và công thức
    for r in range(2, TEMPLATE_ROWS + 2):
        for name, col in header.items():
            if name not in FORMULA_COLS:
                ws.cell(row=r, column=col).value = None
    for i, row in enumerate(rows):
        for name, value in row.items():
            if name in FORMULA_COLS or name not in header:
                continue
            ws.cell(row=2 + i, column=header[name]).value = value

    if seeds is not None:
        wss = wb["Seeds"]
        assert len(seeds) <= 5, "sheet Seeds có 5 ô (A2:A6)"
        for i in range(5):
            wss.cell(row=2 + i, column=1).value = seeds[i] if i < len(seeds) else None

    if summary_notes:
        wsum = wb["Summary"]
        hdr = {cell.value: cell.column for cell in wsum[1] if cell.value}
        col_note = hdr["nhận xét ngắn (bạn viết)"]
        for r in range(2, wsum.max_row + 1):
            g = wsum.cell(row=r, column=1).value
            if g in summary_notes:
                wsum.cell(row=r, column=col_note).value = summary_notes[g]

    wb.calculation.fullCalcOnLoad = True   # Excel/LibreOffice tính lại công thức khi mở file
    wb.save(out_path)
