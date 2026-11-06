# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202603614
- **Họ và tên:** Nguyễn Quang Đạo
- **SEED** (in ở ô đầu notebook): 9526

## 1. Bằng chứng

| Tệp                                                                  | Có? |
| -------------------------------------------------------------------- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output  | ☐   |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☐   |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước          | ☐   |

## 2. Kết quả chính

_Chép bảng so sánh cuối Phần 5 (F1 test, F1 SUB 40, p95 ms/tin, USD/1k tin) của chính bạn._

Bảng so sánh 4 thế hệ AI trên cùng bài toán lọc SMS (`SEED = 9526`, sau khi bỏ 403 tin trùng còn `5.169` tin: `train = 3.619`, `val = 775`, `test = 775`, `SUB = 40`):

| Phương án                                               | Precision | Recall | F1 (test) | F1 (SUB 40) | Train (s) | Tin/s (cả lô) | p50 ms/tin | p95 ms/tin | USD / 1k tin |
| ------------------------------------------------------- | --------: | -----: | --------: | ----------: | --------: | ------------: | ---------: | ---------: | -----------: |
| **1. BoW + LogReg (tự code)**                           |     0.986 |  0.773 |     0.866 |       0.889 |     7.507 |    32,375.728 |      0.063 |      0.132 |        0.000 |
| **2. BoW + MLP (deep learning)**                        |     1.000 |  0.864 |     0.927 |       0.889 |    22.290 |    27,890.901 |      0.373 |      2.619 |        0.000 |
| **3a. Encoder (MiniLM) + kNN**                          |     0.818 |  0.920 |     0.866 |       0.974 |   123.789 |        20.118 |     89.961 |    136.188 |        0.000 |
| **3b. Encoder (MiniLM) + LogReg**                       |     0.908 |  0.898 |     0.903 |       0.947 |   124.231 |        35.998 |     27.666 |     40.167 |        0.000 |
| **3c. Encoder (MiniLM) + MLP head**                     |     0.940 |  0.886 |     0.912 |       0.947 |   125.204 |        35.470 |     28.173 |     40.322 |        0.000 |
| **4a. LLM nhỏ zero-shot** (`openai/gpt-oss-20b`)        |     0.944 |  0.850 |         — |       0.895 |     0.000 |         0.788 |    793.380 |  3,584.323 |      0.000\* |
| **4b. LLM lớn zero-shot** (`openai/gpt-oss-120b`)       |     1.000 |  0.850 |         — |       0.919 |     0.000 |         0.890 |    590.568 |  2,797.581 |      0.000\* |
| **4c. LLM nhỏ few-shot** (`openai/gpt-oss-20b`, 4-shot) |     0.944 |  0.850 |         — |       0.895 |     0.000 |         0.575 |  1,463.985 |  3,430.992 |      0.000\* |

_\*Ghi chú: Phần 4 chạy qua gói miễn phí của Groq (`openai/gpt-oss-20b` và `openai/gpt-oss-120b`) nên chi phí thực trả là $0.00; nếu quy đổi theo đơn giá `gpt-6-luna` ($0.10 / $0.50 mỗi 1M token ở phần tính tay) thì chi phí zero-shot là ~$0.0105 / 1k tin và few-shot tăng ~2.5–3 lần._

## 3. Dự đoán và thực tế

_Chọn 3 dự đoán `predict(...)` lệch nhiều nhất so với kết quả. Mỗi cái: bạn đoán gì, ra gì, vì sao lệch._

1. **`emb_tricky` (Thí nghiệm 3.B — 6 tin nhắn ngụy trang):**
   - **Dự đoán (`14:47:49`):** `Encoder + LogReg` sẽ đoán đúng nhiều hơn hẳn vì `BoW` chỉ đếm từ khoá nên dễ bắt nhầm tin `ham` chứa `"free"`, `"won"`, `"prize"` và bỏ lọt tin `spam` né từ (`"at no cost"`, `"complimentary"`).
   - **Thực tế:** Cả `BoW + LogReg` và `3b. Encoder + LogReg` đều đạt **5/6 câu đúng** (hoà nhau) và cùng đoán sai câu số 6 (_"Last chance! Your account rewards expire tonight, confirm details to keep them"_).
   - **Vì sao lệch:** Ở các câu `ham` có từ `"free"`, `"won"`, mô hình `BoW` không bị lừa nhờ các đại từ giao tiếp cá nhân (`"i"`, `"me"`, `"my"`) mang trọng số âm rất mạnh kéo xác suất về `ham`. Trong khi đó, câu số 6 mang văn phong lừa đảo tài khoản (phishing hiện đại) không xuất hiện trong tập SMS cổ điển (vốn chủ yếu là trúng thưởng/nhạc chuông), khiến cả đầu phân loại `LogReg` gắn trên `MiniLM` cũng dự đoán dưới ngưỡng `0.5`.

2. **`dl_relu` (Thí nghiệm 2.B — Bỏ hàm kích hoạt `nn.ReLU()` trong MLP):**
   - **Dự đoán (`14:42:58`):** Xoá `nn.ReLU()` biến mạng 2 lớp `Linear` liên tiếp thành phép biến đổi tuyến tính tương đương Logistic Regression, nên dự đoán F1 sẽ tụt về gần mức `0.866` của Phần 1 và thấp hơn MLP có ReLU.
   - **Thực tế:** Mạng 2 lớp `Linear(2000 → 64) → Linear(64 → 1)` không có `ReLU` vẫn đạt **F1 = 0.927** trên tập test, ngang bằng với MLP có `ReLU` (`0.927`) và cao hơn `train_logreg` ở Phần 1 (`0.866`).
   - **Vì sao lệch:** Về mặt toán học, tích hai ma trận tuyến tính $W_2(W_1 x + b_1) + b_2$ quả thực biểu diễn cùng không gian giả thuyết tuyến tính như Logistic Regression. Tuy nhiên, `train_mlp` ở Phần 2 dùng bộ tối ưu **Adam** với mini-batch (`batch_size = 64`) và tham số hoá dư thừa (over-parameterization), giúp hội tụ nhanh và tìm được điểm cực tiểu tốt hơn nhiều so với Gradient Descent toàn cục (full-batch GD, `lr = 0.5`, 300 epoch, khởi tạo 0) ở Phần 1.

3. **`llm_models` & `llm_fewshot` (Thí nghiệm 4.A & 4.B — LLM zero-shot và few-shot trên SUB 40):**
   - **Dự đoán (`14:47:51` & `14:49:30`):** Dự đoán LLM sẽ đạt F1 khoảng `0.92 – 0.97` (ngang hoặc nhỉnh hơn `3b. Encoder + LogReg`), và thêm 4 ví dụ mẫu (few-shot) từ tập train sẽ giúp `4c` tăng F1 so với `4a`.
   - **Thực tế:** Trên tập `SUB` (40 tin), `4a. LLM nhỏ zero-shot` đạt **F1 = 0.895**, `4b. LLM lớn zero-shot` đạt **F1 = 0.919**, và `4c. LLM nhỏ few-shot` giữ nguyên **F1 = 0.895** (trong khi độ trễ p50 tăng gần gấp đôi từ `793.38 ms` lên `1,463.98 ms`), đều thấp hơn `3b. Encoder + LogReg` (**F1 SUB = 0.947**) và `3a. Encoder + kNN` (**F1 SUB = 0.974**).
   - **Vì sao lệch:** Tập SMS Spam Collection chứa nhiều tin nhắn viết tắt đặc thù tiếng Anh SMS thập niên 2000 (như _"Did you hear about the new Divorce Barbie..."_ hay _"dating:i have had two of these..."_ được gán nhãn `spam` trong bộ dữ liệu gốc dù nội dung giống tin nhắn đùa/trò chuyện). Các mô hình gắn trên dữ liệu `TRAIN` (`3a`, `3b`, `3c`) học được quy ước gán nhãn đặc thù của bộ dữ liệu này, còn LLM phân loại theo ngữ nghĩa phổ quát nên coi các câu đùa đó là `ham` (recall trên SUB đạt `0.850`).

## 4. Phân tích 50 case khó

_Model nào tụt nhiều nhất, ở nhóm case nào? Nêu 2 case cụ thể và cơ chế khiến model sai._

- **Mô hình tụt mạnh nhất:**
  - `1. BoW + LogReg` tụt mạnh nhất: F1 giảm từ **0.889** (trên SUB) xuống **0.333** (accuracy **52%**), tiếp theo là `2. BoW + MLP` tụt từ **0.889** xuống **0.421** (accuracy **56%**).
  - Nhóm Embedding giữ chất lượng tốt hơn rõ rệt: `3a. Emb + kNN` đạt **F1 = 0.724** (accuracy **68%**), `3b. Emb + LogReg` và `3c. Emb + MLP head` đạt **F1 = 0.640** (accuracy **64%**).
  - Các mô hình `BoW` sụp đổ nặng nhất ở nhóm **Spam giọng tự nhiên (`natural_spam`)** (chỉ đạt **0% – 10%**), **Né từ khoá (`obfuscation`)** (**17%**) và **Thao túng / lừa đảo (`manipulation`)** (**12% – 38%**). Ngược lại, Embedding xử lý tốt nhóm _Né từ khoá_ (**83% – 100%**) nhưng vẫn gặp khó khăn ở nhóm **Nói về spam (`talks_about_spam`)** (**25% – 50%**) và **Spam giọng tự nhiên** (**20% – 40%**). Với LLM zero-shot (`4a`), điểm yếu đặc thù là **Prompt injection**, nhưng khi chuyển sang `4d` (`build_defended_prompt` bọc tin nhắn trong cặp thẻ `<sms>...</sms>` và coi mọi lệnh điều khiển bên trong là dấu hiệu spam) thì chặn được hoàn toàn các ca tấn công chèn lệnh.
- **Phân tích 2 case cụ thể và cơ chế gây sai:**
  1. **Case #21 (`manipulation`, nhãn thật = `spam`):** _"Hi Mum, I dropped my phone in the toilet, this is my new number. Can you send £200 to my friend's account? I'll explain later x"_
     - **Cơ chế sai:** Với `BoW`, câu hoàn toàn không có các từ khoá quảng cáo (`free`, `win`, `txt`, `claim`, `prize`) mà lại chứa nhiều từ hội thoại gia đình (`Hi`, `Mum`, `my`, `phone`, `friend`) mang trọng số âm kéo mạnh về `ham`. Với `Encoder + LogReg`, vector ngữ nghĩa của câu nằm gần cụm hội thoại cá nhân trong tập train (vốn không có mẫu lừa đảo giả danh người thân "Hi Mum" hiện đại), nên đầu phân loại tuyến tính cũng dự đoán nhầm là `ham`.
  2. **Case #30 (`obfuscation`, nhãn thật = `spam`):** _"F R E E e n t r y into our weekly draw, t x t WIN to 80086"_ (hoặc **Case #29**: _"C0ngr4tulat10ns! Y0u h4ve w0n a £1000 g1ft c4rd..."_)
     - **Cơ chế sai:** Hàm `tokenize` dùng regex `[a-z0-9£$€]+` tách `"F R E E"` thành các chữ cái rời `["f", "r", "e", "e"]` và giữ nguyên `"c0ngr4tulat10ns"`, `"w0n"`. Các token biến dạng này không khớp với cột `"free"`, `"congratulations"`, `"won"` trong `vocab` 2.000 từ của `BoW`, làm vector đặc trưng gần như toàn số `0` và khiến `BoW` đoán nhầm thành `ham`.

## 5. Thử thách cá nhân

_Đề theo SEED của bạn, cách làm, kết quả, giải thích._

- **Đề bài theo `SEED = 9526` (`9526 % 6 = 4` → Đề #5):**
  > _Embedding · Viết 4 tin nhắn của riêng bạn (2 spam né từ khoá, 2 ham chứa 'free'/'win'). So dự đoán BoW vs Embedding+LogReg và giải thích từng tin._
- **Cách làm:** Tự thiết kế 4 tin nhắn tiếng Anh (2 tin `spam` dùng từ đồng nghĩa `"complimentary luxury getaway"`, `"at zero cost"`, `"message YES"` để né các từ khoá `free/win/prize/txt`; 2 tin `ham` giao tiếp đời thường nhưng cố tình đưa vào các từ `"free"`, `"win"`, `"call"`, `"now"`, `"prize"`). Đưa cả 4 câu qua `predict_proba` của `1. BoW + LogReg` và `3b. Encoder (MiniLM) + LogReg`.
- **Kết quả thực nghiệm:**

| TT  | Nhãn thật | `BoW` $p(\text{spam})$ | `BoW` đoán | `Emb+LR` $p(\text{spam})$ | `Emb+LR` đoán | Nội dung tin nhắn                                                                            |
| :-: | :-------: | ---------------------: | :--------: | ------------------------: | :-----------: | -------------------------------------------------------------------------------------------- |
|  1  |  `spam`   |                  0.268 |  `ham` ✗   |                     0.961 |   `spam` ✓    | _"Your number was selected for a complimentary luxury getaway, confirm your details today"_  |
|  2  |  `spam`   |                  0.327 |  `ham` ✗   |                     0.969 |   `spam` ✓    | _"Special VIP invitation: receive a brand new tablet at zero cost, message YES to register"_ |
|  3  |   `ham`   |                  0.717 |  `spam` ✗  |                     0.916 |   `spam` ✗    | _"Are you free to call me now? Our team is going to win the prize at tonight's pub quiz!"_   |
|  4  |   `ham`   |                  0.029 |  `ham` ✓   |                     0.292 |    `ham` ✓    | _"I am free tomorrow evening, let us celebrate our big win at the local pub"_                |

- **Giải thích:**
  - **Tin 1 & Tin 2 (`spam` né từ khoá):** `BoW` thất bại cả 2 câu ($p = 0.268$ và $p = 0.327 < 0.5$) vì các cụm _"complimentary luxury getaway"_, _"at zero cost"_, _"message YES"_ né toàn bộ các từ khoá có trọng số dương lớn nhất của `BoW` (`call`, `txt`, `free`, `win`, `prize`). Ngược lại, `Encoder (MiniLM) + LogReg` bắt trúng cả 2 câu với xác suất rất cao ($0.961$ và $0.969$) vì encoder pre-train ánh xạ `"complimentary"` / `"at zero cost"` về cùng vùng không gian ngữ nghĩa với `"free"` và nhận diện đúng cấu trúc thông báo trúng thưởng/dụ đăng ký.
  - **Tin 3 (`ham` chứa dày đặc từ khoá):** Cả `BoW` ($p = 0.717$) và `Emb+LR` ($p = 0.916$) đều bị đánh lừa vì câu tập hợp cùng lúc 5 từ đặc trưng của spam (`free`, `call`, `now`, `win`, `prize`), áp đảo ngữ cảnh buổi đố vui (`"pub quiz"`) ở cuối câu.
  - **Tin 4 (`ham` chứa `'free'` và `'win'`):** Cả `BoW` ($p = 0.029$) và `Emb+LR` ($p = 0.292$) đều đoán đúng là `ham` vì câu không có lời kêu gọi hành động khẩn cấp (`call/txt/claim`) và các từ sinh hoạt đời thường (`"I"`, `"tomorrow"`, `"evening"`, `"local"`) giữ xác suất dưới ngưỡng `0.5`.

## 6. Phương án cho sản phẩm thật

_Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì._

Với bài toán lọc **10 triệu SMS/ngày** (lưu lượng trung bình ~**116 tin/giây**, giờ cao điểm có thể gấp 3–5 lần) và ràng buộc **SLA p95 < 50 ms/tin**:

- **Các phương án thoả mãn SLA (`p95 < 50 ms`):**
  1. `1. BoW + LogReg`: `p95 = 0.132 ms`, thông lượng `32,375.7 tin/s`, `F1 (test) = 0.866`, `F1 (SUB) = 0.889`.
  2. `2. BoW + MLP`: `p95 = 2.619 ms`, thông lượng `27,890.9 tin/s`, `F1 (test) = 0.927`, `F1 (SUB) = 0.889`.
  3. `3b. Encoder (MiniLM) + LogReg`: `p95 = 40.167 ms`, thông lượng `36.0 tin/s`, `F1 (test) = 0.903`, `F1 (SUB) = 0.947`, `F1 (50 case khó) = 0.640`.
  4. `3c. Encoder (MiniLM) + MLP head`: `p95 = 40.322 ms`, thông lượng `35.5 tin/s`, `F1 (test) = 0.912`, `F1 (SUB) = 0.947`, `F1 (50 case khó) = 0.640`.
     _(Các phương án `3a. Encoder + kNN` với `p95 = 136.188 ms` và `4a/4b/4c LLM` với `p95 = 2,797 – 3,584 ms` đều vượt ngưỡng SLA 50 ms; ngoài ra nếu dùng LLM `gpt-6-luna` với chi phí $10.50 / 1 triệu tin thì 10 triệu tin/ngày sẽ tốn **$105/ngày ≈ $3,150/tháng**)._
- **Phương án lựa chọn & đánh đổi:**
  - **Nếu triển khai 1 mô hình duy nhất:** Chọn **`3b. Encoder (MiniLM) + LogReg`** (hoặc `3c`). Mô hình này nằm trên **đường Pareto**, vừa đạt chuẩn SLA (`p95 = 40.17 ms < 50 ms`), vừa có chất lượng vượt trội trên tập `SUB` (`F1 = 0.947`) và bền vững hơn nhiều trước spam biến thể/né từ khoá trên bộ 50 case khó (`F1 = 0.640` so với `0.333 – 0.421` của BoW), chi phí API bằng `$0`. Đánh đổi: tốc độ encode từng lô trên 1 tiến trình CPU đạt `36 tin/s` nên cần chạy **4–6 worker/instance song song** (hoặc dùng 1 GPU nhỏ / ONNX quantization) để đáp ứng lưu lượng `> 116 tin/s`.
  - **Kiến trúc tối ưu cho sản phẩm thật (Hệ thống lai — Cascade Router 3 tầng):**
    - **Tầng 1 (`2. BoW + MLP` kết hợp chuẩn hoá leetspeak):** Xử lý tức thời (`p95 = 2.62 ms`, `~27,890 tin/s`, precision = `1.000` trên tập test) cho ~**85% – 90%** tin nhắn có xác suất rất rõ ràng ($p < 0.05$ hoặc $p > 0.95$).
    - **Tầng 2 (`3b. Encoder MiniLM + LogReg`):** Chỉ chạy cho ~**10% – 14%** tin nhắn vùng biên từ Tầng 1, giúp bắt spam đồng nghĩa/né từ khoá mà vẫn giữ `p95` toàn hệ thống dưới `10 ms` và giảm 85% tải tính toán cho Encoder.
    - **Tầng 3 (`4d. LLM + prompt phòng thủ <sms>`):** Chỉ kiểm tra bất đồng bộ (hoặc lấy mẫu ~**0.5% – 1%** tin lưỡng lự nhất có $p \in [0.4, 0.6]$) để phát hiện chiến dịch lừa đảo/thao túng mới và tự động gán nhãn bổ sung dữ liệu huấn luyện định kỳ cho Tầng 1 & 2 (chi phí API giảm hơn 99%, chỉ còn ~$15 – $30/tháng).

## 7. AI log và điều đã học

_Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu._

### Nhật ký sử dụng AI (AI Log)

| Phần                        | Bạn hỏi AI gì (prompt)                                                                                                                                                                          | AI trả lời / viết gì                                                                                                                                                                                | Bạn đã sửa / kiểm chứng gì                                                                                                                                                                         |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Phần 4 (TODO 4.2 – 4.3)** | Hỏi cách dùng biểu thức chính quy (regex) để loại bỏ khối suy luận `<think>...</think>`, xử lý các cụm phủ định (`not spam`, `non-spam`, `not ham`) và tách nhãn ở đầu câu trong `parse_label`. | AI gợi ý dùng `re.sub` xoá thẻ `<think>`, chuẩn hoá dấu nháy (`n't` → `not`), đảo nghĩa cụm `not/non + ham/spam` trước khi khớp nhãn ở đầu chuỗi hoặc kiểm tra tập nhãn duy nhất bằng `re.findall`. | Tự đọc hiểu từng bước regex, kiểm tra với các trường hợp biên (`"Not spam"`, `"Label: spam"`, `"I am not sure"`) và chạy `_run_check("4.2–4.3 prompt / parse_label", _t)` đạt ✅.                  |
| **Phần 4 (TODO 4.1 & 4.4)** | Hỏi cách truyền `**provider_kwargs(max_tokens)` vào `client.chat.completions.create` trong `call_openai` và công thức ước tính chi phí token trong `llm_classify_batch`.                        | AI giải thích cách unpack `provider_kwargs` để tương thích với `max_completion_tokens` / `reasoning_effort` của từng nhà cung cấp và cách tính `tokens = len(words) / 0.75`.                        | Tự hoàn thiện hàm `call_openai` và `llm_classify_batch`, kiểm tra xử lý `n_invalid` khi `parse_label` trả về `None` và chạy `_run_check("4.1 + 4.4 call_openai / llm_classify_batch", _t)` đạt ✅. |

### Điều đã học hôm nay

Trước buổi lab, em từng nghĩ mô hình càng lớn và hiện đại (như LLM hàng chục/trăm tỷ tham số) thì sẽ luôn thắng áp đảo các mô hình cổ điển trên mọi tiêu chí. Qua thực nghiệm trực tiếp cả 4 thế hệ AI trên cùng một bài toán, em hiểu ra rằng **biểu diễn đặc trưng (representation) và sự khớp phân phối dữ liệu** mới là yếu tố quyết định: một encoder nhỏ gọn như `MiniLM` kết hợp `Logistic Regression` (`3b`) có thể đạt F1 cao hơn cả LLM zero-shot trên tập kiểm thử trong khi nhanh hơn gần 70 lần và hoàn toàn miễn phí. Đồng thời, bài kiểm tra 50 case khó giúp em nhận thức rõ điểm mù của từng phương pháp (`BoW` mù ngữ cảnh và dễ vỡ vụn trước leetspeak, `Embedding` dễ nhầm khi người dùng kể lại chuyện spam, còn `LLM` dễ bị tấn công _prompt injection_ nếu không đóng khung dữ liệu bằng thẻ phân cách), từ đó thấy được vì sao hệ thống thực tế luôn cần thiết kế dạng **Router đa tầng** thay vì phụ thuộc vào một mô hình duy nhất.
