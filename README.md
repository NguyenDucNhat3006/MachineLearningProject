# AirPulse V4 — cơ hội giảm giá và quyết định kiểm tra lại

Model nghiên cứu chạy CPU, kèm FastAPI, dataset, model huấn luyện sẵn và kiểm thử. Mục tiêu là ước lượng **trong một khoảng thời gian có cơ hội giảm giá đủ đáng kể không**, sau đó quyết định mua hoặc kiểm tra lại.

**Đọc `THAY_DOI_V4.md` để xem kết quả thực tế và những giới hạn chưa sửa được bằng code.** Dữ liệu phần lớn là bán tổng hợp; chế độ live chưa được xác nhận. API trả trạng thái rõ ràng khi chưa đủ điều kiện tư vấn.

## Chạy trên Windows / PowerShell

Yêu cầu Python 3.11 trở lên; khuyến nghị dùng môi trường mới. Không cần CUDA hoặc GPU.

```powershell
Expand-Archive .\AirPulse_V4_Booking_Model_Package.zip -DestinationPath .
cd .\AirPulse_V4_Booking_Model
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe demo.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Mở `http://127.0.0.1:8000/docs`, thử `POST /insight` bằng `examples/request_replay.json`. `POST /predict` là alias cùng schema.

Ví dụ gọi từ cửa sổ PowerShell khác:

```powershell
$requestBody = Get-Content .\examples\request_replay.json -Raw -Encoding utf8
Invoke-RestMethod -Uri http://127.0.0.1:8000/insight -Method Post -ContentType 'application/json' -Body ([System.Text.Encoding]::UTF8.GetBytes($requestBody))
```

## Linux / macOS

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python demo.py
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

## Huấn luyện và tái lập

Đã có `trained/airpulse_v4_model.joblib`; không bắt buộc train lại để thử API.

```bash
python train.py
python forward_diagnostic.py
python demo.py
python -m unittest discover -s tests -v
```

- `train.py`: chia nguồn thành train / calibration / policy-validation / test, train các nhánh giá tại mốc, cơ hội giảm, mức tiết kiệm cơ hội và chi phí kỳ vọng ngắn hạn; hiệu chuẩn, chọn blend và policy bằng các tập dành riêng, lưu model trước khi tính metric test.
- `forward_diagnostic.py`: chẩn đoán nội bộ với nguồn tách biệt và thời gian có embargo; rất ít nguồn test nên không xem là xác nhận thị trường.
- Mặc định `python train.py` tái tạo model, CSV quyết định, bảng thử chính sách và manifest.
- Nếu file đầu vào không có `source_flight_id` hợp lệ hoặc không đủ cửa sổ có nhãn, chương trình dừng thay vì chia dòng tùy ý.
- Model test và dataset cũ đã được xem trong quá trình phát triển. Bộ benchmark này không phải test thị trường mới, dù code chọn tham số không dùng bảng kết quả test để tối ưu.

Khi có dataset mới đúng schema, tạo split mới **trước khi train**:

```bash
python make_source_split.py --data data/my_real_quotes.csv --out data/my_source_split.csv
python train.py --data data/my_real_quotes.csv --source-split data/my_source_split.csv --out my_trained
```

Tiện ích yêu cầu tối thiểu 30 nhóm nguồn để bố trí các tập riêng; đây chỉ là điều kiện cho workflow, không phải số mẫu đảm bảo hiệu quả. Các cờ research/live trong model và script vẫn giữ bảo thủ; kiểm chứng thị trường và provenance cần được bổ sung theo dự án.

## Hợp đồng API

Request replay hoàn chỉnh nằm trong `examples/request_replay.json`, gồm `mode=replay`, `as_of` và một mảng `history` chứa các object báo giá. Có thể dùng trực tiếp file đó trong Swagger hoặc PowerShell; không cần tự điền dữ liệu giả.

Mỗi báo giá: `itinerary_id`, `observed_at`, `departure_at`, `origin`, `destination`, `airline`, `price_vnd`; tùy chọn `currency=VND`, `fare_family`, `seller`, `cabin`, `baggage`. Không gửi nguyên mọi cột CSV vào API: schema từ chối trường ngoài hợp đồng để không âm thầm bỏ qua điều kiện quan trọng. Tất cả timestamp phải có múi giờ. Chỉ gửi các quan sát đã biết tại `as_of`.

Các trường chính trong response:

| Trường | Ý nghĩa |
|---|---|
| `status` | `OK` hoặc một trạng thái từ chối tư vấn có mã |
| `recommendation.action` | `BUY_NOW`, `WAIT_AND_RECHECK`, hoặc `null` khi chưa đủ cơ sở |
| `recommended_buy_at` | Bằng `as_of` khi mua ngay; `null` khi chờ |
| `next_check_at` | Lần kiểm tra tiếp theo khi chờ; không phải giờ đáy được đảm bảo |
| `forecast_slots[].prob_drop_within_window` | Xác suất có giá thấp hơn trong cả cửa sổ; cần giảm cả ≥4% và ≥50.000đ |
| `forecast_price_vnd` | Ước lượng tại cuối horizon, có fallback persistence |
| `estimated_opportunity_low_vnd` | Mức thấp cơ hội được ước lượng; không phải giá mua được hoặc giá live |
| `empirical_90pct_lower_vnd/upper_vnd` | Khoảng từ sai số calibration; không có bảo đảm độ phủ ngoài thị trường |
| `policy_validation` | Cho biết policy WAIT được validation chấp thuận hay đang tắt |
| `warnings` | Tính mô phỏng, điều kiện vé chưa xác minh và giới hạn dự báo |

`BUY_NOW` với lý do `NO_VALIDATED_WAIT_ADVANTAGE` nghĩa là chưa xác nhận lợi thế của việc chờ theo chính sách đã kiểm tra; **không có nghĩa giá chắc chắn không giảm**.

Giới hạn mặc định: ít nhất 4 timestamp phân biệt, lịch sử trải dài ít nhất 12 giờ, báo giá cuối không cũ quá 6 giờ, dự báo 6/12/24/48/72 giờ, hạn mua trước khởi hành 36 giờ. Chu kỳ kiểm tra lại 3 giờ, dung sai thực thi trễ tối đa 3,1 giờ. Khi backtest chỉ dùng báo giá **ở hoặc sau** thời điểm kiểm tra lại, tuyệt đối không lấy giá trước đó để giả vờ đã mua được.

## Replay và live

- Replay bắt buộc `as_of`, dùng để xem lại dữ liệu quá khứ hoặc kịch bản mô phỏng được gắn nhãn.
- Live dùng giờ server, không cho client đặt `as_of`. Chuyến đã bay, báo giá cũ và tuyến ngoài phạm vi có trạng thái riêng.
- Model trong gói có `validated_for_live=false`. Một request live hợp lệ vẫn nhận `NOT_VALIDATED_FOR_LIVE`, không có hành động mua/chờ. Muốn dùng live cần thay bằng model được kiểm chứng trên dữ liệu thị trường mới và kiểm tra tổng giá, người bán/điều kiện vé; không chỉ đổi cờ để làm mất cảnh báo.
- Lịch sử báo giá trong API mẫu do client cung cấp để thử nghiệm. Khi triển khai, server nên truy xuất lịch sử chuẩn từ kho của hệ thống theo ID chuyến và điều kiện vé.

## Kết nối frontend

`integration/modelApi.js` cung cấp hàm gọi `/insight` và ánh xạ trạng thái V4 sang UI. Đây là adapter theo hợp đồng mới, không phải source website Vercel hiện tại. Cần truyền chuỗi báo giá thật/replay phù hợp và `asOf`; không thể chỉ truyền tuyến/ngày bay rồi suy ra lịch sử.

CORS mặc định cho phép `https://airpulse-demo.vercel.app`, `http://localhost:5173`, `http://127.0.0.1:5173`. Có thể đặt `AIRPULSE_CORS_ORIGINS` thành danh sách origin phân cách bằng dấu phẩy; tránh dùng wildcard tùy ý. `AIRPULSE_MODEL` chọn đường dẫn model nếu cần.

Nên hiển thị nhãn replay, thời điểm quan sát cuối, trạng thái thiếu dữ liệu; phân biệt cơ hội giảm trong khoảng với giá dự báo tại mốc. Không ánh xạ giá dự báo thành “Fair Price” hoặc báo giá thanh toán. Schema OpenAPI kèm trong `examples/openapi.json`.

## Cấu trúc

- `airpulse_v4.py`: dữ liệu, đặc trưng, nhãn, suy luận và backtest tuần tự.
- `train.py`: train, calibration, chọn policy và benchmark.
- `forward_diagnostic.py`: kiểm tra nội bộ tách nguồn + thời gian.
- `api.py`, `integration/modelApi.js`: API hai endpoint và adapter frontend.
- `demo.py`, `examples/`: request/response mẫu có thời điểm rõ ràng.
- `tests/`: kiểm thử lỗi và hành vi quan trọng.
- `data/`: ba CSV gốc giữ nguyên, kèm split V3.
- `trained/`: model, manifest, chỉ số, bảng chọn policy và các quyết định.
- `legacy/`: code/model V3 dùng đối chiếu; không dùng thay model V4 trong API.
- `THAY_DOI_V4.md`: thay đổi, kết quả, giới hạn và những việc cần làm trên website.

Nguồn gốc dữ liệu: gói V3 do người dùng cung cấp. Các ngày tương lai trong BASE/STRESS tiếp tục được ghi rõ là mô phỏng. Gói V4 không bổ sung quan sát thị trường mới.
