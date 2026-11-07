# AirPulse V4 — thay đổi, cách chạy và kết quả kiểm chứng

Ngày đóng gói: **21/10/2026**. Nguồn đầu vào: gói AirPulse V3 do bạn cung cấp. Phiên bản phần mềm: `4.0.0`; schema `airpulse.insight.v4`.

## 1. Bản này cải thiện điều gì?

V4 là một bản triển khai mới của pipeline dự báo cơ hội giảm giá, kiểm tra chất lượng đầu vào và ra quyết định có kiểm tra lại. Gói đã có model train sẵn, FastAPI, adapter frontend, notebook Colab, các CSV gốc, dữ liệu kiểm chứng và 26 kiểm thử đã qua.

**Chưa thể gọi đây là model mua vé rẻ nhất hoặc model đã chứng minh tiết kiệm tiền tốt hơn V3.** Trên dữ liệu hiện tại, bộ chọn chính sách chỉ dựa vào policy-validation chưa thấy lợi thế chờ đủ vững, nên bản phát hành giữ `policy.enabled=false`. Các đầu ra học máy về cơ hội giảm giá và khoảng giá vẫn có, nhưng action mặc định quay về mua ngay theo baseline khi đủ điều kiện replay; trạng thái thiếu dữ liệu thì không có action.

Đây là kết quả lựa chọn được công bố, không phải một model hoàn hảo luôn tìm đúng đáy.

## 2. Các thay đổi đã triển khai

| Vấn đề V3 | Thay đổi trong V4 | Bằng chứng/đường dẫn |
|---|---|---|
| Nhãn giảm tại một mốc chưa trả lời cơ hội trong khoảng | Tạo `y_drop_within`: tồn tại báo giá trong cửa sổ giảm cả ≥4% và ≥50.000đ | `supervised_pairs()`; test giá giảm rồi hồi phục |
| Dễ gán nhãn không giảm khi thiếu tương lai | Loại cửa sổ bị right-censor hoặc thiếu độ phủ, không gán thành negative | `supervised_pairs()`; test censored window |
| Chi phí chờ chưa có tín hiệu riêng | Thêm model thay đổi giá kỳ vọng trong 6h; các policy ứng viên có thể yêu cầu mức tiết kiệm kỳ vọng tối thiểu | `short_term_cost_model`; test economic gate |
| Giá dự báo chưa vượt persistence | Model log-return L1 có lựa chọn blend trên calibration; yêu cầu cải thiện ít nhất 1% để rời baseline | `calibration_point_blend.csv`, metadata model |
| Xác suất chưa hiệu chuẩn | Hiệu chuẩn sigmoid trên nhóm nguồn calibration riêng, không phải train | `calibrator`, bảng calibration trong metrics |
| Chỉ train/val/test | Tách 4 phần nguồn: train, calibration, policy-validation, test | `source_partitions.csv` |
| Một nguồn sinh nhiều chuỗi dễ chi phối | Trọng số huấn luyện cân bằng theo nguồn gốc | `source_weights()` |
| Tháng/năm giả lập dễ bị hiểu thành mùa vụ thật | Đặc trưng V4 bỏ tháng, năm và thứ trong tuần; giữ thời gian tới ngày bay, giờ và lịch sử đã quan sát | `features` trong manifest |
| Thiếu lag bị coi như không đổi giá | Thêm cờ thiếu lag 24h/72h | `missing_lag_24h`, `missing_lag_72h` |
| One-shot không thực thi recheck | Mô phỏng kiểm tra lại, quyết định mới và mua ở quan sát sau thời điểm chờ | `sequential_backtest()` |
| Chỉ một thời điểm bắt đầu | Đánh giá tại thời điểm đủ lịch sử đầu tiên và các mốc khoảng 2/4/6/8 ngày trước bay khi có dữ liệu | `entry_id`, `entry_days_to_departure` |
| Dùng giá cũ nếu không có giá sau khi chờ | Đánh dấu không thực thi được, chi phí để trống; không quay về quá khứ mua ngay | Test `backtest_never_buys_in_the_past...` |
| Chọn mốc trước khi kiểm tra điều kiện | Lọc mốc đủ điều kiện rồi mới xếp hạng | `decide()`; test eligible slot |
| Chọn policy theo trung bình có thể quá lạc quan | Đối chiếu baseline, downside và lower bound bootstrap theo nguồn; yêu cầu ≥3 nguồn có WAIT | `validation_policy_grid.csv` |
| Không phân biệt replay/live | Replay bắt buộc `as_of`; live dùng giờ server và chưa được bật với model chưa kiểm chứng | `recommend()`, `api.py` |
| Bốn dòng chỉ vài giây vẫn dự báo | Yêu cầu ≥4 timestamp và lịch sử trải dài ≥12h | Test short history |
| Giá cũ, chuyến đã bay, tuyến lạ vẫn BUY | Có `STALE_QUOTES`, `DEPARTURE_PASSED`, `OUT_OF_SCOPE`, action null | Core/API tests |
| Lẫn điều kiện vé | Kiểm tra nhất quán seller, cabin, fare family, baggage, tiền tệ nếu được truyền | Test mixed seller/fare |
| BUY NOW kèm mốc mua tương lai | `recommended_buy_at=as_of` khi BUY; WAIT chỉ trả `next_check_at` | Test BUY/WAIT timestamps |
| API `/predict` khác `/insight` trên web | Cả hai endpoint dùng chung schema V4 | Test both routes |
| Thiếu CORS khi gọi khác origin | CORS có allowlist, đã thử cả origin được phép và không được phép | API tests |
| Trường ngoài schema bị bỏ qua | Từ chối trường ngoài hợp đồng, timestamp thiếu múi giờ, giá không hữu hạn | Pydantic + core tests |
| Khó theo dõi phiên bản/model/dataset | Có ID, model/data/code hash, dependency versions, settings và split manifest | `trained/manifest.json` |

## 3. Kiến trúc model

Các nhánh LightGBM:

1. **Opportunity classifier:** xác suất có mức giảm đáng kể trong 6/12/24/48/72 giờ.
2. **Opportunity magnitude regressor:** mức tiết kiệm cơ hội từ giá nhỏ nhất quan sát trong khoảng. Đây là nhãn tương lai để học; không phải giá được đảm bảo mua.
3. **Point-price regressor:** log-return tại cuối cửa sổ, L1 và blend/persistence guard.
4. **Short-term expected-cost regressor:** biến động giá kỳ vọng 6h để policy có thể tránh chờ qua một đợt tăng kỳ vọng.

Một sigmoid calibrator hiệu chuẩn nhánh phân loại. Policy kiểm tra điều kiện, downside và bằng chứng validation để bật/tắt WAIT. Không phải reinforcement learning đã tìm được nghiệm dừng tối ưu; đây là một policy thực nghiệm được kiểm chứng bằng mô phỏng tuần tự.

**Giá trị cơ hội thấp nhất trong tương lai chỉ có vai trò nhãn hoặc oracle hậu kiểm. Nó không có trong features lúc dự báo.** Test đã kiểm tra việc đổi giá tương lai không làm thay đổi đặc trưng quá khứ.

## 4. Dữ liệu và cách chia

- BASE: 65.067 quan sát, 720 hành trình bán tổng hợp, 217 nguồn gốc.
- REAL: 21.057 quan sát của 232 chuyến, cùng ngày bay thật 14/10/2026 theo nhãn trong gói gốc.
- STRESS: cùng nguồn/lịch mô phỏng, thay giá theo giả định. Không dùng để nhân đôi train.
- Train: **514 hành trình / 151 nguồn**.
- Calibration: **48 / 16 nguồn**.
- Policy-validation: **65 / 17 nguồn**.
- Test: **93 / 33 nguồn**.

Các CSV gốc được giữ nguyên byte; chương trình đóng gói đã đối chiếu hash với gói V3. Calibration và policy-validation là hai nhóm nguồn tách riêng.

Các chỉ số test là benchmark phát triển được tái sử dụng: dữ liệu đã được xem trong đánh giá V3 và quá trình xây V4. Hàm chọn policy không đọc kết quả test, nhưng điều đó không biến tập đã xem thành một test thị trường hoàn toàn mới. Bộ sinh dữ liệu gốc dùng thống kê toàn nguồn và chưa có code trong gói đầu vào; hạn chế này vẫn còn.

## 5. Kết quả thực tế — không chỉ liệt kê thay đổi code

### 5.1. Dự báo giá trên cùng trường hợp

So sánh hai bản trên **cùng 5,247 cặp giá–horizon** của giao thức V4:

| Phương án | MAE |
|---|---:|
| V3 dự báo trên các đầu vào tương ứng | **190.835đ** |
| V4 | **159.670đ** |
| Giữ nguyên giá hiện tại | **159.670đ** |

MAE V4 thấp hơn V3 khoảng **16.33%** trong phép so sánh này. **Nguyên nhân là guard đã chọn persistence vì nhánh giá học được chưa vượt baseline đủ mạnh trên calibration.** Không nên diễn giải kết quả này là nhánh ML giá của V4 đã học được khả năng dự báo vượt persistence.

Không so trực tiếp số 159.670đ này với 262.118đ của báo cáo V3 cũ: hai giao thức có cửa sổ/nhãn/sampling khác nhau. Bảng trên dùng cùng trường hợp để tránh so lệch.

### 5.2. Nhánh cơ hội giảm trong khoảng

| Chỉ số V4 test | Giá trị |
|---|---:|
| AUC sự kiện giảm trong cửa sổ | 0.7629 |
| Average precision | 0.3089 |
| Brier | 0.10622 |
| Tỷ lệ sự kiện trong tập | 13.68% |
| Độ phủ thực nghiệm của khoảng giá ghi nhãn 90% | 89.88% |

AUC này không so trực tiếp với AUC 0,6967 của V3 vì sự kiện gán nhãn đã khác. Khoảng giá dùng sai số calibration; không có bảo đảm độ phủ ngoài thị trường hoặc ở các dịp Tết chưa có quan sát thật.

### 5.3. Mô phỏng mua/chờ tuần tự

Test có **321 tình huống bắt đầu** trên **65 hành trình / 26 nguồn** đủ độ phủ tương lai. Các điểm bắt đầu trên cùng hành trình phụ thuộc nhau; không phải 321 chuyến độc lập. CSV loại trừ ghi rõ các tình huống thiếu dữ liệu.

| Phương án, trên cùng các tình huống đủ độ phủ | Tiết kiệm trung bình so với mua ngay |
|---|---:|
| Mua ngay | 0đ |
| V4 phát hành, WAIT đang tắt | **0đ** |
| Policy V3 được kiểm tra lại mỗi 3h, cùng hạn mua | **8.984đ** |

**V4 chưa vượt V3 về tiền tiết kiệm trong benchmark tuần tự này.** Bộ bảo vệ của V4 chọn không chờ vì policy-validation chưa cung cấp bằng chứng đủ vững. Không được quảng bá lợi ích tiền bạc chỉ bằng việc sai số giá thấp hơn.

V4 có 0 lệnh WAIT mặc định; CI tiết kiệm [0,0] chỉ phản ánh đang dùng baseline mua ngay, không chứng minh model luôn đúng. Các dự báo cơ hội vẫn có thể hiển thị như thông tin nghiên cứu. Muốn policy WAIT được bật, cần train/validation mới đủ tốt; không chọn một ngưỡng khác dựa trên việc nhìn test thấy lời.

V3 trong bảng được chạy lại theo giao thức recheck mới, không phải one-shot gốc âm 5.108đ/vé. Vì vậy không dùng hai số khác giao thức để tạo cảm giác V4 đã cải thiện lợi nhuận.

### 5.4. Kiểm tra thời gian + nguồn

Có thêm `forward_diagnostic.py`, với các giai đoạn thời gian không chồng lấn và giữ nhóm nguồn độc lập:

- Train: 238 hành trình.
- Calibration: 9.
- Policy-validation: 14.
- Test: 11 hành trình / 11 nguồn trước lọc độ phủ.
- AUC nội bộ: 0.7139; MAE: 163.439đ.

Đây vẫn là dữ liệu bán tổng hợp đã dùng trong phát triển. Số nguồn ít và nguồn tạo dữ liệu dùng thống kê toàn bộ gốc; không xem phép kiểm tra này là chứng nhận thị trường mới.

## 6. Cách dùng ngay

Giải nén ZIP, mở thư mục `AirPulse_V4_Booking_Model`. Trên Windows dùng Python 3.11+:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe demo.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Mở `http://127.0.0.1:8000/docs`, thử `/insight` bằng `examples/request_replay.json`. Không cần train lại để chạy demo. Notebook `AirPulse_V4_Colab.ipynb` có hướng dẫn upload ZIP, demo, train tùy chọn và kiểm thử CPU.

Huấn luyện lại gói hiện tại:

```bash
python train.py
python forward_diagnostic.py
python demo.py
```

Khi có dữ liệu mới, xem `make_source_split.py` và phần hướng dẫn trong README. File cần source ID gốc, timestamp có múi giờ, cùng điều kiện vé và đủ cửa sổ quan sát. Ngưỡng số nhóm nguồn của utility chỉ để tạo được workflow, không bảo đảm chất lượng.

## 7. Thông số và ý nghĩa response

- Tối thiểu 4 timestamp, lịch sử trải dài 12h; tuổi báo giá tối đa 6h.
- Horizon 6/12/24/48/72h; hạn mua mặc định trước bay **36h** để công bố rõ phạm vi hỗ trợ.
- Kiểm tra lại 3h; cho phép báo giá xuất hiện trễ tối đa 3,1h. Khoảng dư 0,1h là grace cho lịch lấy mẫu có jitter, không cho phép dùng giá trước thời điểm recheck.
- Không có giá thực thi sau khi chờ: trạng thái không thực thi được và để trống cost, không quay lại mua ở giá trước đó.
- `BUY_NOW`: `recommended_buy_at=as_of`; không phải cam kết đã tìm đáy.
- `WAIT_AND_RECHECK`: `next_check_at` là lần kiểm tra tiếp, không phải giờ mua rẻ nhất được bảo đảm.
- Trạng thái thiếu dữ liệu hoặc ngoài phạm vi: action `null`.
- `forecast_price_vnd` và mức thấp cơ hội là ước lượng, không phải giá bán còn mua được.
- Live dùng giờ server và bị chặn bởi `NOT_VALIDATED_FOR_LIVE` với model nghiên cứu này. Không chỉ đổi cờ để công bố model đã được kiểm chứng.

## 8. Kiểm thử và tái lập

**26/26 kiểm thử qua** trên source và model cuối. Các nhóm gồm: dip rồi hồi phục, feature không đọc tương lai, cửa sổ censored, thời gian ngắn/cũ/tương lai, tuyến ngoài phạm vi, hỗn hợp điều kiện vé, nonfinite price, timestamp BUY/WAIT, thiếu báo giá thực thi, nhiều thời điểm vào, nhóm nguồn tách biệt, API hai endpoint, CORS allowlist, schema và trạng thái research.

Kết quả nằm trong `trained/test_results.txt`, `test_summary.json`, `metrics.json`, `manifest.json`, CSV quyết định và các bảng policy. `SHA256SUMS.json` ghi hash các file trong ZIP. Các dependency lõi được pin theo môi trường đã chạy; không cần GPU/CUDA.

## 9. Những việc vẫn cần dữ liệu hoặc source frontend

1. Có giá thật nhiều ngày khởi hành, nhiều thời điểm và điều kiện vé/người bán nhất quán; giữ toàn bộ snapshots.
2. Kiểm chứng trên một đợt thị trường mới chưa được dùng trong phát triển. Dữ liệu gốc một ngày khởi hành không đủ xác nhận hiệu ứng Tết/hè/năm 2027.
3. Kết nối frontend vào `/insight` bằng history đúng schema; adapter ở `integration/modelApi.js`. Gói model không tự thay đổi bản deploy Vercel.
4. Đưa nhãn replay/chuyến đã bay/độ mới giá lên UI; phân biệt chưa có dữ liệu mới với giá thực sự không đổi; sửa lời hứa về phí và nội dung Fair Price.
5. Thay phần client gửi báo giá bằng server lấy lịch sử đã chuẩn hóa trong sản phẩm thật, kiểm tra giá checkout và khả năng còn vé.

Model và pipeline tốt hơn về định nghĩa cơ hội, kiểm tra đầu vào, tính trung thực của phép đánh giá và khả năng tích hợp. **Kết quả hiện có chưa đủ chứng minh policy mua/chờ tốt hơn V3 hoặc tốt hơn mua ngay trên thị trường.**
