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


def history_rows(results: list[dict]) -> list[dict]:
    """Lịch sử từng epoch của mọi lần chạy, dạng dài (một dòng = một exp_id × một epoch) cho sheet "History"."""
    keys = ("train_loss", "train_acc", "train_loss_running", "val_loss", "val_acc", "val_macro_f1",
            "grad_norm", "grad_norm_max", "frac_clipped", "lr", "epoch_time_s")
    out = []
    for r in results:
        h = r["history"]
        for i, ep in enumerate(h["epoch"]):
            out.append({"exp_id": r["cfg"]["exp_id"], "epoch": ep,
                        **{k: _num(h[k][i]) if k in h and i < len(h[k]) else None for k in keys}})
    return out


def _write_sections(ws, sections: list[tuple[str, list[dict]]]) -> None:
    """Ghi lần lượt các bảng (tiêu đề in đậm, dòng tên cột, các dòng số) xuống một sheet, cách nhau một dòng trống."""
    from openpyxl.styles import Font
    r = 1
    for title, table in sections:
        if title:
            ws.cell(row=r, column=1, value=title).font = Font(bold=True)
            r += 1
        cols = list(dict.fromkeys(k for row in table for k in row))
        for j, name in enumerate(cols, 1):
            ws.cell(row=r, column=j, value=name).font = Font(bold=True)
        for row in table:
            r += 1
            for j, name in enumerate(cols, 1):
                v = row.get(name)
                v = v.item() if hasattr(v, "item") else v      # số numpy -> số Python
                ws.cell(row=r, column=j, value=_num(v) if isinstance(v, float) else v)
        r += 2


def write_xlsx(rows: list[dict], template_path: str, out_path: str,
               seeds: list[str] | None = None, summary_notes: dict[str, str] | None = None,
               extra_sheets: dict[str, list[tuple[str, list[dict]]]] | None = None) -> None:
    """Điền các dòng vào sheet "Experiments" của mẫu (từ dòng 2), sheet "Seeds" (exp_id baseline khác seed)
    và cột nhận xét của sheet "Summary", rồi lưu thành out_path. Giữ nguyên mọi công thức của mẫu.

    extra_sheets: {tên sheet mới: [(tiêu đề bảng, các dòng dict), ...]} cho số liệu ngoài bảng thí nghiệm
    (phép thử sức khoẻ, khởi tạo, phân tích theo lớp, lịch sử từng epoch...) để mọi số trong báo cáo đều tra lại được."""
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

    for name, sections in (extra_sheets or {}).items():
        _write_sections(wb.create_sheet(name), sections)

    wb.calculation.fullCalcOnLoad = True   # Excel/LibreOffice tính lại công thức khi mở file
    wb.save(out_path)
    fill_cached_values(out_path)


# ----------------------------------------------------------------------------------------------- giá trị cache
# openpyxl chỉ ghi công thức, không ghi kết quả (<v> rỗng). Excel/LibreOffice tự tính lại khi mở, nhưng
# pandas.read_excel hay openpyxl(data_only=True) sẽ thấy ô trống. Hàm dưới tính lại đúng các công thức của
# mẫu (bằng Python, từ chính các ô nhập trong file) và ghi kết quả vào <v> để mọi trình đọc đều thấy số.

def _blank(x):
    return x is None or x == ""


def _computed_values(path: str) -> dict[str, dict[str, object]]:
    """{tên sheet: {ô: giá trị}} cho mọi ô công thức của mẫu; "" là chuỗi rỗng như Excel trả về."""
    wb = openpyxl.load_workbook(path)
    ws, wss, wsum = wb["Experiments"], wb["Seeds"], wb["Summary"]
    col = {c.value: c.column_letter for c in ws[1] if c.value}
    rows = range(2, TEMPLATE_ROWS + 2)
    get = lambda name, r: ws[f"{col[name]}{r}"].value
    out = {"Experiments": {}, "Seeds": {}, "Summary": {}}

    # Seeds: tra exp_id -> val_acc (B), val_macro_f1 (C), best_val_loss (D); rồi mean / std mẫu / 2σ
    by_id = {get("exp_id", r): r for r in rows if not _blank(get("exp_id", r))}
    seeds_cols = {"B": "val_acc", "C": "val_macro_f1", "D": "best_val_loss"}
    for L, name in seeds_cols.items():
        vals = []
        for r in range(2, 7):
            e = wss[f"A{r}"].value
            v = "" if _blank(e) or e not in by_id or _blank(get(name, by_id[e])) else get(name, by_id[e])
            out["Seeds"][f"{L}{r}"] = v
            if isinstance(v, (int, float)):
                vals.append(float(v))
        mean = sum(vals) / len(vals) if vals else ""
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) if len(vals) >= 2 else ""
        out["Seeds"].update({f"{L}8": mean, f"{L}9": std, f"{L}10": "" if std == "" else 2 * std})
    base_f1, noise = out["Seeds"]["C8"], out["Seeds"]["C10"]

    # Experiments: 4 cột công thức ở cuối
    for r in rows:
        p, s, t, v = (get(n, r) for n in ("step0_loss", "final_train_loss", "final_val_loss", "val_macro_f1"))
        af = "" if _blank(v) or base_f1 == "" else v - base_f1
        out["Experiments"].update({
            f"{col['step0_gap_vs_lnC']}{r}": "" if _blank(p) else p - math.log(7),
            f"{col['gap_val_minus_train']}{r}": "" if _blank(t) or _blank(s) else t - s,
            f"{col['delta_val_f1_vs_base']}{r}": af,
            f"{col['beyond_noise']}{r}": "" if af == "" or noise == "" else ("Có" if abs(af) > noise else "Không"),
        })

    # Summary: đếm / max / min theo group
    for r in range(2, 12):
        g = wsum[f"A{r}"].value
        grp = [rr for rr in rows if get("group", rr) == g]
        f1s = [get("val_macro_f1", rr) for rr in grp if not _blank(get("val_macro_f1", rr))]
        accs = [get("val_acc", rr) for rr in grp if isinstance(get("val_acc", rr), (int, float))]
        n_f1 = len(f1s)
        out["Summary"].update({
            f"B{r}": len(grp), f"C{r}": n_f1,
            f"D{r}": "" if n_f1 == 0 else max(f1s), f"E{r}": "" if n_f1 == 0 else min(f1s),
            f"F{r}": "" if n_f1 == 0 else (max(accs) if accs else 0),
        })
        if str(wsum[f"G{r}"].value).startswith("="):
            out["Summary"][f"G{r}"] = "Có" if n_f1 > 0 else "Chưa"
    out["Summary"]["D13"] = sum(out["Summary"].get(f"G{r}") == "Có" for r in range(2, 12))

    # mọi ô công thức của mẫu phải có giá trị (mẫu đổi bố cục thì báo lỗi thay vì ghi sai)
    for name, vals in out.items():
        formulas = {c.coordinate for row in wb[name].iter_rows() for c in row
                    if isinstance(c.value, str) and c.value.startswith("=")}
        assert formulas == set(vals), f"{name}: ô công thức không khớp {sorted(formulas ^ set(vals))[:5]}"
    return out


def fill_cached_values(path: str) -> None:
    """Ghi kết quả của các ô công thức vào <v> trong XML của file xlsx (công thức giữ nguyên)."""
    from xml.sax.saxutils import escape
    import re
    import zipfile

    values = _computed_values(path)
    sheet_files = {n: f"xl/worksheets/sheet{i + 1}.xml" for i, n in enumerate(openpyxl.load_workbook(path).sheetnames)}
    with zipfile.ZipFile(path) as z:
        entries = [(info, z.read(info.filename)) for info in z.infolist()]

    def patch(xml: str, vals: dict) -> str:
        def repl(m):
            ref, attrs, formula = m.group(1), m.group(2), m.group(3)
            if ref not in vals:
                return m.group(0)
            v = vals[ref]
            attrs = re.sub(r'\s+t="[^"]*"', "", attrs)
            if isinstance(v, str):
                return f'<c r="{ref}"{attrs} t="str"><f>{formula}</f><v>{escape(v)}</v></c>'
            return f'<c r="{ref}"{attrs}><f>{formula}</f><v>{repr(float(v)) if isinstance(v, float) else v}</v></c>'
        return re.sub(r'<c r="([A-Z]+\d+)"([^>]*)><f>(.*?)</f>(?:<v\s*/>|<v>.*?</v>)?</c>', repl, xml)

    tmp = path + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for info, data in entries:
            for name, fname in sheet_files.items():
                if info.filename == fname and name in values:
                    data = patch(data.decode("utf-8"), values[name]).encode("utf-8")
            z.writestr(info, data)
    Path(tmp).replace(path)
