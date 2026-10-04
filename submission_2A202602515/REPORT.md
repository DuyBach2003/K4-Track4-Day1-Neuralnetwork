# Báo cáo Lab Day 1 — Đoàn Duy Bách — 2A202602515

Mọi con số trỏ về một `exp_id` trong `experiments.xlsx` (và `results/<exp_id>.json`). Số theo từng epoch nằm ở sheet `History`; số ngoài bảng thí nghiệm (phép thử sức khoẻ, std kích hoạt khi khởi tạo, val so với eval, phân tích theo lớp, thống kê dữ liệu) nằm ở sheet `Extra`. "Vượt nhiễu" nghĩa là chênh lệch val macro-F1 so với trung bình baseline lớn hơn 2σ = **0,0203** (mục 2). Mục 3 chỉ dùng **val**; eval chỉ xuất hiện ở mục 4.

## 1. Thiết lập

- **Môi trường:** Apple M5, GPU Apple qua **MPS** (không có CUDA), Python 3.12.14, PyTorch 2.14.1. `code/lab.ipynb` chạy Restart & Run All trong 12,9 phút. Pipeline tất định: `base-s1` trùng từng epoch với `opt-sgdm-lr0.3` (cùng cấu hình, cùng seed).
- **Dữ liệu:** `train` 464 809 / `eval` 116 203 theo `split_metadata.csv`. Validation là 20% của train (phân tầng, seed 42), còn lại 371 847 train / 92 962 val. Chuẩn hoá 10 cột số bằng mean/std của phần train còn lại; 44 cột nhị phân giữ nguyên.
- **Model và baseline:** `M-base` (54→256→128→7, 47 879 tham số, có `assert`); CE, SGD+momentum 0,9, **lr = 0,3** (chọn bằng val, mục 3.2), batch 512, 20 epoch, He (`kaiming_normal_`, bias 0), không dropout, không clip, FP32.
- **Mốc:** accuracy "đoán lớp đa số" trên val = **0,4876**.
- **Chủ đề đã thử:** ☑ loss ☑ optimizer ☑ hyper-parameter ☑ dropout ☑ clipping ☑ mixed precision ☑ init. Có **52 lần chạy**, mỗi lần một dòng trong bảng và một ảnh `figures/<exp_id>.png`, cộng 14 ảnh `compare_*.png`.
- **Cách đo:** best epoch = epoch có val loss thấp nhất, metric lấy ở epoch đó. Train loss đo ở eval mode trên toàn bộ train. `grad_norm` đo trước khi clip. `time_per_epoch_s` lấy trung bình từ epoch 2, vì epoch 1 gồm thời gian biên dịch kernel. MPS không có `max_memory_allocated` nên `peak_mem_MB` là giá trị lấy mẫu, đã **gồm 124 MB dữ liệu** nằm sẵn trên GPU.

## 2. Kiểm tra ban đầu và độ nhiễu

| Kiểm tra | Kết quả |
|---|---|
| Số tham số / shape logits | 47 879 / (B, 7) |
| Loss bước 0 (so với ln 7 = 1,946) | **2,269** (`base-s1`); 1,90–2,31 qua 5 seed |
| Quá khớp 20 mẫu: loss cuối | 2,198 → < 1e-6 sau 500 bước (Adam 1e-2), accuracy 100% |
| Mọi tham số có gradient khác 0 | ☑ có (‖grad‖ của W1…b3 từ 0,27 đến 1,94) |
| Baseline, số seed đã chạy | 5 (`base-s1` … `base-s5`) |
| Baseline: val acc (TB ± σ) | 0,9122 ± 0,0026 |
| Baseline: val macro-F1 (TB ± σ) | 0,8591 ± 0,0101 |

**Ngưỡng nhiễu: 2σ = 0,0203 (val macro-F1).** Ước lượng từ 5 seed nên còn thô. Macro-F1 nhiễu gấp ~4 lần accuracy vì phụ thuộc vào lớp hiếm (lớp 3 chỉ có ~440 mẫu val).

Loss bước 0 cao hơn ln 7 khoảng 0,3 nhưng **không phải lỗi**. He nhân phương sai với 2 để bù cho ReLU, mà lớp cuối không có ReLU, nên logit có std ≈ 0,6 thay vì ≈ 0. Đối chứng trong notebook: đặt W3 = 0 thì loss ra đúng **1,9459**. Các init cho logit nhỏ (`init-normal`, `init-zeros`) cũng có loss bước 0 = 1,946.

Đường cong baseline (`base-s1.png`, `compare_baseline.png`): train và val loss cùng giảm và vẫn còn dốc ở epoch 20 (best epoch = 20 ở 4/5 seed). Val − train chỉ 0,025, tức mô hình **chưa khớp hết và chưa quá khớp**.

## 3. Kết quả theo chủ đề

### 3.1 Hàm mất mát: CE vs MSE (`compare_loss.png`)
- **Dự đoán:** gradient theo logit của CE là (p − y)/B. MSE (`F.mse_loss(logits, one_hot)`, trung bình trên B×7 phần tử, không có ½) cho 2(z − y)/(7B). Ở z ≈ 0, logit lớp đúng nhận −0,86/B với CE so với −0,29/B với MSE, nên MSE học chậm hơn ~3 lần. Mình đoán MSE kém quá 2σ, và tăng lr ×3 sẽ thu hẹp được một phần.
- **Kết quả:** `loss-mse` đạt F1 **0,7843** (−0,075, vượt nhiễu), acc 0,883. Ngay epoch 1 F1 chỉ 0,533 so với 0,675 của CE; ‖g‖ trung bình nhỏ hơn ~5 lần (0,067 so với 0,357). `loss-mse-lr0.9` (đổi 2 yếu tố) còn tệ hơn: **0,6879**.
- **Cơ chế:** MSE kéo mọi logit về 0/1, kể cả logit lớp sai ở mẫu đã đúng, và gradient không tăng khi mô hình sai một cách tự tin. Vì vậy ranh giới của lớp hiếm được tối ưu kém. Phần tăng lr khác dự đoán: ở lr 0,9 có gai ‖g‖ = 10,8 ở epoch 1, sau đó F1 kẹt ~0,20 vài epoch. Mình *phỏng đoán* (chưa đo) đây là ReLU chết hàng loạt, giống `clip-none-lr3`. Không so giá trị loss (0,027 so với 0,226) vì hai loss khác thang đo.

### 3.2 Bộ tối ưu hoá (`compare_optimizer.png`, `compare_optimizer_lr.png`)
- **Dự đoán:** Adam/AdamW ở lr 1e-3–3e-3 tốt nhất và vượt SGD+momentum quá 2σ; SGD thuần cần lr gấp ~10 lần; Adam ≈ AdamW (wd 0,01).
- Mỗi bộ được so ở lr tốt nhất của nó (lưới 3–5 lr, cách nhau ~3 lần). lr tốt nhất đều nằm trong lưới, trừ SGD thuần nằm ở mép lr = 1:

| Bộ tối ưu | exp_id | lr | val macro-F1 | best epoch | Δ so với TB baseline |
|---|---|---|---|---|---|
| Adam (β = 0,9/0,999, ε = 1e-8) | `opt-adam-lr0.003` | 3e-3 | **0,8719** | 18 | +0,013 (trong nhiễu) |
| AdamW (wd = 0,01) | `opt-adamw-lr0.003` | 3e-3 | 0,8646 | 18 | +0,006 (trong nhiễu) |
| SGD + momentum 0,9 | `opt-sgdm-lr0.3` | 0,3 | 0,8571 | 20 | −0,002 |
| SGD | `opt-sgd-lr1` | 1 | 0,8232 | 15 | −0,036 (vượt nhiễu) |

- **Độ nhạy lr:** SGD+momentum tụt từ 0,857 xuống 0,773 khi lr tăng từ 0,3 lên 1, và chỉ còn 0,764 ở lr 0,01. Adam phẳng hơn quanh đỉnh (0,843 / 0,872 / 0,861 ở 1e-3 / 3e-3 / 1e-2). `opt-adamw-wd0-lr0.003` trùng `opt-adam-lr0.003` đến từng bit.
- **Cơ chế:** Adam chia bước cho √v̂ của từng tham số nên xuống nhanh ở đầu (val loss epoch 3: 0,339 so với 0,365). Sau 20 epoch, SGD+momentum đã chỉnh lr gần như bắt kịp. **Adam hơn nhưng chưa vượt nhiễu**, nên chưa kết luận được Adam thắng (dự đoán sai). Momentum cộng dồn các hướng gradient nhất quán (bước hiệu dụng ≈ η/(1−μ)), nên SGD thuần thua rõ dù dùng lr = 1, mức đã bắt đầu có gai ‖g‖ = 9,2.

### 3.3 Hyper-parameter (`compare_hparam.png`, `compare_hparam_batch.png`)
Mọi dòng giữ SGD+momentum lr 0,3 và chỉ đổi một yếu tố. Số bước mỗi epoch là 2 906 / 727 / 182 cho batch 128 / 512 / 2048.

| exp_id | thay đổi | s/epoch | val F1 | Δ | vượt nhiễu? |
|---|---|---|---|---|---|
| `hp-bs128` | batch 128 | 1,21 | 0,7958 | −0,063 | có |
| `hp-bs2048` | batch 2048 | 0,10 | 0,8399 | −0,019 | không (sát) |
| `hp-bs2048-lrx4-warmup` | batch 2048, lr 1,2, khởi động 1 epoch | 0,10 | 0,8611 | +0,002 | không |
| `hp-wide` | 54-512-256-7 | 0,36 | 0,8736 | +0,015 | không |
| `hp-deep` | 54-256-128-64-7 | 0,37 | 0,8537 | −0,005 | không |
| `hp-ep40` | 40 epoch | 0,31 | 0,8757 | +0,017 | không |
| `hp-cosine` | lr cosine → 0 | 0,31 | **0,8885** | **+0,030** | **có** |
| `hp-wd5e-4` | weight decay 5e-4 | 0,32 | 0,7076 | −0,152 | có |

- **Batch:** batch 128 có số bước gấp 4 lần, nhưng vẫn giữ lr 0,3 nên phương sai mỗi bước (∝ η²/B) tăng ~4 lần. Val loss dao động và F1 giảm 0,063 (trái dự đoán). Batch 2048 cùng lr thì thiếu bước (−0,019). Áp **quy tắc tăng lr theo lô** (lr ×4 + khởi động) đưa nó về ngang baseline, với số bước ít hơn 4 lần và mỗi epoch nhanh hơn 3 lần.
- **Lịch lr cosine** là thay đổi có lợi nhất. Cùng số bước, lr giảm dần về 0 làm giảm nhiễu của SGD ở cuối, cho val loss thấp nhất nhóm (0,168 so với 0,226). M-wide và 40 epoch có val loss tốt hơn (0,197 và 0,202) nhưng F1 chưa vượt 2σ, nên dự đoán "vượt 2σ" của mình quá lạc quan.
- **Weight decay 5e-4** ở lr 0,3 co trọng số theo (1 − ηλ) mỗi bước, tích luỹ qua 14 540 bước ≈ e^−2,2. Mô hình vốn đã chưa khớp nên bị đẩy sang chưa khớp nặng (train loss 0,396).

### 3.4 Dropout (`compare_dropout.png`)
- **Dự đoán:** baseline chưa quá khớp (val − train = 0,025), nên dropout chỉ làm giảm năng lực và F1 sẽ giảm.
- **Kết quả:** F1 giảm đơn điệu theo q: 0,8391 (q 0,1, −0,020, sát ngưỡng) → 0,7899 (q 0,3) → 0,6475 (q 0,5). Hai mức sau vượt nhiễu, khớp dự đoán. Khoảng cách val − train thu hẹp (0,025 → 0,012 → 0,007 → 0,004) **vì train loss tăng** (0,201 → 0,414), chứ không phải vì val tốt lên. Loss lúc train (có dropout) cao hơn loss đo ở eval mode (q 0,5: 0,507 so với 0,414), đó là lý do phải đo train loss ở eval mode khi vẽ.

### 3.5 Gradient clipping (`compare_clipping.png`)
- **Ở lr bình thường:** ‖g‖ của baseline rất đều (p5–p95 = 0,29–0,44), chỉ có một gai 2,87 ở bước đầu. Mình chọn c = 0,35 (trung vị) để clipping thật sự kích hoạt. `clip-0.35` cắt 68% số bước nhưng F1 0,8617 (+0,003, trong nhiễu): khi không có bất ổn, clipping chỉ làm bước nhỏ lại.
- **Ở lr cao (phản chứng):** với lr ×3 = 0,9, không clip thì có gai ‖g‖ = **10,0** và F1 0,7928. Có clip thì ‖g‖ lớn nhất còn 2,99 và F1 **0,8263 (+0,034, vượt nhiễu)**, trong khi chỉ cắt 20% số bước ở epoch 1 và 1–2% về sau. Với lr ×10 = 3 thì **cả hai đều hỏng** (khác dự đoán). Không clip có gai 163, sau đó val loss kẹt ở 1,21 ≈ entropy phân phối lớp (1,205): mạng đoán toàn lớp 1 vì ReLU chết. Loss không thành NaN nên cờ `diverged` không bật. Có clip thì F1 chỉ 0,193, vì clipping chặn được gradient nhưng với momentum 0,9 bước cập nhật vẫn dài tới ηc/(1−μ) ≈ 10.

### 3.6 Mixed precision (`compare_amp.png`, `compare_amp_xwide.png`)

| exp_id | s/epoch | peak mem (MB) | phần do huấn luyện (MB) | val F1 |
|---|---|---|---|---|
| `base-s1` (FP32) | 0,31 | 131,1 | 4,3 | 0,8571 |
| `amp-fp16` (+ GradScaler) | 0,64 | 131,2 | 4,4 | 0,8467 |
| `amp-bf16` | 0,35 | 131,2 | 4,4 | 0,8566 |
| `amp-xwide-fp32` (2048-2048, 5 epoch) | 7,55 | 187,6 | 60,8 | 0,8313 |
| `amp-xwide-fp16` | 7,54 | 192,7 | 65,9 | 0,8266 |
| `amp-xwide-bf16` | 5,21 | 192,7 | 65,9 | 0,8345 |

- **Dự đoán và kết quả:** độ chính xác như nhau (trong nhiễu) và M-base không nhanh hơn: **FP16 chậm gấp ~2,1 lần, BF16 chậm ~14%**. Mỗi bước chỉ có phép nhân ma trận cỡ 512×256, nên thời gian do chi phí gọi kernel quyết định. Autocast thêm kernel ép kiểu, GradScaler thêm bước unscale/kiểm tra inf (bỏ 4 bước do tràn số khi s = 65 536 còn lớn). Ở mạng 2048-2048 (chỉ so trong nhóm), phép nhân ma trận chiếm ưu thế nên BF16 nhanh hơn 31%; FP16 thì ngang FP32 (7,54 so với 7,55 s), và thời gian của FP16 dao động giữa các lần chạy nên mình không kết luận FP16 nhanh hơn. Bộ nhớ lại tăng thêm 5 MB vì autocast giữ thêm bản FP16 của trọng số.
- **Vì sao FP16 cần nhân loss với s:** FP16 chỉ có 5 bit mũ (số dương nhỏ nhất ~6·10⁻⁸) nên gradient nhỏ bị làm tròn về 0. BF16 có 8 bit mũ như FP32 nên gần như không underflow, chỉ kém chính xác hơn. Không đo được trên GPU NVIDIA, nên kết luận về tốc độ chỉ đúng cho MPS.

### 3.7 Khởi tạo tham số (`compare_init.png`, `compare_init_depth.png`)

| init | std h1 / h2 / logit (bước 0) | loss bước 0 | val F1 (exp_id) |
|---|---|---|---|
| he | 0,390 / 0,366 / 0,577 | 2,269 | 0,8571 (`base-s1`) |
| xavier (`xavier_normal_`, 2/(n_vào+n_ra)) | 0,163 / 0,125 / 0,192 | 2,022 | 0,8534 (`init-xavier`) |
| default (U(±1/√n), bias ngẫu nhiên) | 0,160 / 0,068 / 0,059 | 1,983 | 0,8638 (`init-default`) |
| normal N(0; 0,01²) | 0,020 / 0,0022 / 0,0003 | 1,946 | 0,8653 (`init-normal`) |
| zeros | 0 / 0 / 0 | 1,946 | **0,0936** (`init-zeros`) |

- **zeros hỏng đúng như dự đoán:** ‖grad‖ bước 0 bằng 0 ở W1, b1, W2, b2, W3; chỉ b3 có gradient (0,497). Kích hoạt ẩn = ReLU(0) = 0 nên ∂L/∂W3 = (p−y)h₂ᵀ = 0, và tín hiệu ngược W3ᵀ(p−y) cũng bằng 0. Mạng chỉ học được tần suất lớp: acc 0,4876, val loss 1,206 = entropy phân phối lớp.
- Các init còn lại **trong nhiễu so với He** ở mạng 3 lớp; normal chỉ chậm ở epoch 1 (F1 0,619 so với 0,675). Mạng 3 lớp không đủ sâu để thấy khác biệt. Trên mạng 30 lớp × 256 ở bước 0, He giữ std ~0,4 → 0,25, còn xavier giảm theo cấp số nhân tới 4,6·10⁻⁶ (với ReLU, mỗi lớp nhân phương sai với ½). normal(0,01) về 0 sau lớp 20, đúng như hình "30 lớp ReLU" của slide.

## 4. Đánh giá cuối trên tập eval

Cấu hình cuối được chọn chỉ bằng val, theo quy tắc viết trước trong notebook (mục 3.8): lấy bộ tối ưu có val F1 cao nhất, rồi chỉ thêm những kỹ thuật hơn baseline **quá 2σ**. Kết quả là **Adam lr 3e-3 + lịch lr cosine**, M-base, batch 512, 20 epoch, He, FP32. M-wide và 40 epoch bị loại vì chưa vượt 2σ. Bài nộp dùng **seed 1** (cố định từ trước), trọng số ở best epoch (= 20).

| Cấu hình | Seed nộp | val macro-F1 | **eval macro-F1** | eval accuracy |
|---|---|---|---|---|
| Baseline (`base-s1`) | 1 | 0,8571 | **0,8586** | 0,9089 |
| Cấu hình cuối cùng (`final-s1`) | 1 | 0,8839 | **0,8868** | 0,9238 |

- **Cải thiện trên eval:** +0,0281 macro-F1, +0,0148 accuracy. Trên val, cấu hình cuối đạt 0,8856 ± 0,0015 (3 seed) so với 0,8591 ± 0,0101 (5 seed): chênh +0,027, vượt 2σ của baseline. Mình không chạy eval cho các seed khác nên không có σ trên eval; vì val và eval gần nhau, mình dùng σ của val làm ước lượng.
- **Val và eval rất gần nhau:** +0,0016 (baseline) và +0,0029 (cấu hình cuối), khớp mốc ≤ 0,005, nên val là ước lượng đáng tin của eval.
- **Nói thẳng:** `hp-cosine` (SGD+momentum + cosine, 1 seed) đạt 0,8885, ngang cấu hình cuối. Nghĩa là **lợi ích gần như hoàn toàn đến từ lịch lr cosine**; Adam không thêm gì đo được.

### 4.1 Phân tích lỗi theo lớp (`final-s1`, eval)

| Lớp | Loại | support | precision | recall | F1 | F1 baseline |
|---|---|---|---|---|---|---|
| 0 | Spruce/Fir | 42 368 | 0,9266 | 0,9133 | 0,9199 | 0,9073 |
| 1 | Lodgepole Pine | 56 661 | 0,9281 | 0,9426 | 0,9353 | 0,9239 |
| 2 | Ponderosa Pine | 7 151 | 0,9135 | 0,9245 | 0,9190 | 0,8935 |
| 3 | Cottonwood/Willow | 549 | 0,8664 | 0,8033 | 0,8336 | 0,8019 |
| 4 | Aspen | 1 899 | 0,8684 | 0,7751 | **0,8191** | 0,7684 |
| 5 | Douglas-fir | 3 473 | 0,8502 | 0,8336 | 0,8418 | 0,8019 |
| 6 | Krummholz | 4 102 | 0,9447 | 0,9327 | 0,9387 | 0,9136 |

Ma trận nhầm lẫn (hàng = thật, cột = dự đoán):

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| **0** | 38 694 | 3 435 | 0 | 0 | 32 | 10 | 197 |
| **1** | 2 761 | 53 407 | 150 | 0 | 181 | 135 | 27 |
| **2** | 2 | 165 | 6 611 | 44 | 8 | 321 | 0 |
| **3** | 0 | 2 | 74 | 441 | 0 | 32 | 0 |
| **4** | 46 | 349 | 20 | 0 | 1 472 | 12 | 0 |
| **5** | 10 | 161 | 382 | 24 | 1 | 2 895 | 0 |
| **6** | 248 | 27 | 0 | 0 | 1 | 0 | 3 826 |

- **Lớp khó nhất là lớp 4 (Aspen), F1 = 0,819.** Nó hay bị nhầm thành **lớp 1 (Lodgepole Pine): 349/1 899 = 18,4%**.
- **Lý giải bằng dữ liệu (thống kê trên X_tr):** Aspen chỉ có 6 075 mẫu train (1,6%), so với 181 312 mẫu của Lodgepole, nên ranh giới bị kéo về phía lớp đông. Độ cao của Aspen (p10–p90: 2 674–2 904 m) nằm gọn trong khoảng 2 664–3 165 m của Lodgepole, và cả hai cùng ở vùng hoang dã 0 và 2. Tương tự, lớp 3 bị nhầm sang lớp 2 13,5% (cùng vùng hoang dã 3, độ cao 2 080–2 350 m nằm trong 2 112–2 639 m), và lớp 5 bị nhầm sang lớp 2 11,0% (độ cao gần trùng: 2 149–2 657 so với 2 112–2 639 m). Cấu hình cuối cải thiện mọi lớp, nhiều nhất ở lớp hiếm (lớp 4 +0,051, lớp 5 +0,040, lớp 3 +0,032).
- **Sẽ thử:** trọng số lớp trong CE (∝ 1/√tần suất) hoặc lấy mẫu cân bằng cho lớp 3/4/5, chọn bằng val macro-F1.

## 5. Trả lời các câu hỏi dẫn dắt

1. **Bộ tối ưu nào thắng khi chỉnh lr công bằng?** Adam (0,8719) > AdamW (0,8646) > SGD+momentum (0,8571), nhưng các chênh lệch này **không vượt 2σ**, nên chỉ kết luận được là SGD thuần thua rõ (0,8232). Nếu không chỉnh lr, chẳng hạn cùng lr 0,01, Adam (0,861) hơn SGD+momentum (0,764) gần 0,1, một kết luận sai do lr.
2. **Dropout có giúp khi chưa quá khớp không?** Không: F1 giảm đơn điệu theo q (q 0,5: −0,21) vì dropout giảm năng lực của một mạng vốn đang thiếu năng lực. Nên dùng khi train loss thấp hơn val rõ rệt và val loss bắt đầu tăng.
3. **Gradient clipping giải quyết gì?** Các gai gradient đột ngột. Ở lr 0,9, gai 10,0 bị chặn còn 2,99 và F1 tăng 0,034 (`clip-none-lr0.9` so với `clip-0.35-lr0.9`), trong khi chỉ 1–2% số bước bị cắt sau epoch 1. Clipping không cứu được lr quá lớn (lr 3) và không có tác dụng khi huấn luyện vốn ổn định.
4. **Mixed precision có nhanh hơn không?** Trên M-base thì không (FP16 ×2,1 thời gian, BF16 +14%), vì chi phí gọi kernel áp đảo và autocast thêm phép ép kiểu. Chỉ khi phép nhân ma trận chiếm ưu thế (2048-2048) BF16 mới nhanh hơn rõ (31%); FP16 ở đó chỉ ngang FP32.
5. **Vì sao init zeros hỏng? He khác Xavier ở đâu?** Tính đối xứng cộng với ReLU(0) = 0 làm gradient của mọi trọng số bằng 0, chỉ bias lớp cuối học được tần suất lớp. He dùng Var = 2/n_vào, Xavier dùng 2/(n_vào+n_ra) ≈ 1/n_vào; hệ số 2 bù cho việc ReLU bỏ nửa phân phối. Khác biệt này quan trọng khi mạng sâu (30 lớp: xavier teo còn 10⁻⁶), không đáng kể với 3 lớp.
6. **Loss không giảm sau 2 000 bước: 3 phép kiểm tra đầu tiên.**
   1. **So loss bước 0 với ln C.** Lệch nhiều là dấu hiệu sai ở lớp cuối hoặc ở chuẩn hoá. Nếu loss *đứng yên* ở đúng entropy phân phối lớp (≈ 1,21 ở đây) thì mạng chỉ còn học bias; mình gặp đúng triệu chứng này ở `init-zeros` và `clip-none-lr3`.
   2. **Quá khớp một lô nhỏ (20 mẫu)** với mọi chính quy hoá tắt. Nếu loss không về ≈ 0 thì lỗi nằm ở code (nhãn, softmax hai lần, `zero_grad`, tham số không vào optimizer), không phải ở dữ liệu hay năng lực mô hình.
   3. **In ‖grad‖ theo từng lớp và `grad_norm` theo bước.** Bằng 0 (`init-zeros`) nghĩa là gradient không chảy hoặc ReLU chết. Có gai lớn rồi sụp (`clip-none-lr3`: gai 163) nghĩa là lr quá cao: giảm lr 3–10 lần hoặc clip. Đặt train loss cạnh val loss để phân biệt chưa khớp với quá khớp.

## 6. Hạn chế và điều bất ngờ

- **Khác dự đoán:** batch 128 tệ hơn (do giữ nguyên lr); MSE với lr ×3 tệ hơn; clipping không cứu được lr ×10; Adam không vượt nhiễu so với SGD+momentum đã chỉnh lr; init normal(0,01) không chậm đáng kể với 3 lớp; weight decay 5e-4 hại nặng hơn nhiều so với dự đoán; lr tốt nhất của SGD+momentum là 0,3 chứ không phải 0,03–0,1.
- **Những gì có thể làm kết luận sai:** (i) 2σ đo trên baseline, giả định các cấu hình khác có độ nhiễu tương tự, trong khi mọi thí nghiệm khác chỉ chạy 1 seed; (ii) cùng số epoch nhưng khác số bước khi đổi batch, và lr không chỉnh lại cho batch 128 hay khi thêm weight decay; (iii) mỗi lần chỉ đổi một yếu tố nên không đo được tương tác (M-wide + cosine + 40 epoch có thể tốt hơn cấu hình cuối); (iv) thời gian đo trên MPS, không phải GPU NVIDIA.
- **Thay đổi giữa chừng (có ghi trong notebook):** lần chạy đầu bị ngắt trước Part 4, khi chưa chạm tới eval. Sau đó mình mở rộng lưới lr (SGD+momentum thêm 1, Adam/AdamW thêm 1e-2, vì lr tốt nhất nằm ở mép) và thêm mức lr ×3 cho clipping. Các ô dự đoán gốc giữ nguyên, kèm ghi chú bổ sung.
- **Nếu có thêm thời gian:** chạy 3 seed cho các thí nghiệm sát ngưỡng (`hp-wide`, `hp-ep40`, `drop-0.1`, Adam so với SGD+momentum); thử M-wide + cosine + 40 epoch; dùng trọng số lớp cho lớp hiếm.

## 7. Phụ lục

- **File nộp:** `REPORT.md`, `experiments.xlsx` (52 dòng; sheet Seeds = `base-s1..5`; Summary có nhận xét; thêm sheet `Extra` và `History`), `predictions_eval.csv` (116 203 dòng, từ `final-s1`), `eval_result.json`, `figures/` (52 ảnh `<exp_id>.png` + 14 ảnh `compare_*.png`), `results/` (52 file JSON + dự đoán/kết quả eval của baseline), `code/` (`lab.ipynb` và 6 module, `requirements.txt`).
- **Thời gian chạy:** toàn bộ notebook 12,9 phút trên Apple M5 (MPS); một lần chạy baseline 20 epoch ≈ 11 giây (0,31 s/epoch huấn luyện, phần còn lại là đánh giá train + val sau mỗi epoch).
