# Chuẩn bị bài nộp

Chưa có checkpoint, log huấn luyện, báo cáo PDF và slide. Thư mục này mới có quy trình chuẩn bị; chưa phải bộ bài nộp hoàn chỉnh. Các tệp lớn để trong `nop-bai/output/` (đã bỏ qua trên git).

## Trước khi đánh giá cuối

- Lấy checkpoint 1-1 và màn thưởng đã được nhóm chọn, chỉ dùng trọng số do nhóm tự huấn luyện.
- Đối chiếu từng checkpoint với commit đã ghi trước lần chạy, `config.yaml`, seed, số bước và ghi chú của chủ lần chạy. Giữ lại cấu hình và log của cả giai đoạn trước nếu có resume hoặc học chuyển giao.
- Trên đúng commit của từng lần chạy, chạy toàn bộ `python -m pytest -q` trong môi trường Python 3.10–3.12 đã cài đủ thư viện. Ghi kết quả và phiên bản thư viện vào hồ sơ nộp.
- Kiểm tra cấu hình nhúng trong checkpoint khớp `config.yaml`, đúng tập hành động, kiến trúc mạng và số frame stack. Với lần chạy cũ, dùng phiên bản mã tương ứng; không sửa cấu hình để ép nạp trọng số.
- Giữ nguyên `mario_ppo/evaluate.py` và chế độ đánh giá của `env.py`.

## Đánh giá theo đề

Tạo thư mục đầu ra trước; thay đường dẫn checkpoint bằng tệp đã chọn:

```powershell
New-Item -ItemType Directory -Force nop-bai/output/danh-gia
python -m mario_ppo.evaluate --ckpt "<checkpoint-1-1.pt>" --level 1-1 --episodes 20 --seed 2026 --out nop-bai/output/danh-gia/ket-qua-1-1.json --video nop-bai/output/danh-gia/1-1.mp4
python -m mario_ppo.evaluate --ckpt "<checkpoint-man-thuong.pt>" --level "<man-thuong>" --episodes 20 --seed 2026 --out "nop-bai/output/danh-gia/ket-qua-<man-thuong>.json" --video "nop-bai/output/danh-gia/<man-thuong>.mp4"
```

Giữ mặc định `max_steps=3000`, `threshold=0.8` và chính sách lấy mẫu của script. JSON phải có `episodes=20`, `seed=2026`, đúng màn và bước checkpoint; các danh sách `x_pos`, `steps`, `noop_steps` phải dài 20. Ghi `flag_rate` thực tế kể cả khi dưới 0.8. Mở video để kiểm tra tệp phát được; không dùng video từ đánh giá 5 lượt trong lúc huấn luyện thay cho video cuối.

Không chạy đánh giá trên checkpoint chưa được chọn rồi gắn tên thành kết quả cuối. Nếu đổi checkpoint, chạy lại cả JSON và video tương ứng.

## Nội dung tệp nén

Tạo một thư mục `nop-bai/output/nhom-<so>/` chứa:

| Thư mục / tệp | Nội dung |
|---|---|
| `ma-nguon/` | Mã Python, tests, cấu hình, `pyproject.toml`, README, phân công; không chứa `.git`, môi trường Python hay cache |
| `cac-lan-chay/<ten-lan-chay>/` | `config.yaml`, `progress.csv`, `evals.csv`, checkpoint được chọn và ghi chú nguồn gốc |
| `danh-gia/` | JSON và video theo giao thức cuối cho 1-1 và màn thưởng |
| `bao-cao.pdf` | Bản báo cáo cuối tối đa 8 trang |
| `trinh-chieu/` | Tệp trình chiếu cuối của nhóm |
| `tai-tao.md` | Commit mã, môi trường thư viện, phần cứng, seed, lệnh chạy, số bước, thời gian, sps và chuỗi checkpoint cha nếu học chuyển giao |

Nếu các lần chạy dùng các commit khác nhau, giữ thêm bản mã đúng commit của mỗi lần trong `ma-nguon-theo-lan-chay/`. Có thể xuất mã bằng `git archive --format=zip --output=<duong-dan.zip> <commit>`; lệnh này lấy tệp đã commit và không mang theo `.git` hay checkpoint trong `runs/`.

Giữ cả log và cấu hình của những thí nghiệm được dùng trong báo cáo, nhất là E0 và E2. Không chép nguyên tất cả lần chạy thử, thư viện Python hoặc video trung gian vào bài nộp.

## Kiểm tra trước khi gửi

1. Bảng pytest, kết quả đánh giá và số liệu trong báo cáo đều lấy từ lần chạy thực; tên màn, checkpoint và seed nhất quán giữa các tệp.
2. Đường dẫn trong lệnh tái tạo dùng được sau khi giải nén. Tệp checkpoint cha của lần học chuyển giao cũng có trong bộ nộp.
3. Thử nạp checkpoint và chạy lại lệnh đánh giá từ bản mã đã đóng gói, trong một thư mục giải nén riêng. So lại seed, NOOP, số lượt và kết quả; ghi môi trường máy nếu có sai khác số học.
4. Đủ bảng cấu hình, sáu đường cong học, thí nghiệm so sánh, phân tích thất bại và phân công trong báo cáo. TV3 đọc soát, TV2 ghép bản PDF cuối.
5. Nén thư mục đã kiểm tra thành `nhom-<so>.zip`, mở lại tệp nén để kiểm tra đủ nội dung. Chỉ tạo bản nộp cuối khi đủ các đầu ra trên.

Đánh giá cuối, kiểm tra checkpoint và đóng gói thực tế vẫn chờ dữ liệu của nhóm; chưa có kết quả nào được xác nhận ở các bước này.
