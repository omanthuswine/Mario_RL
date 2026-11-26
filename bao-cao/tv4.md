# Phần báo cáo TV4

## Cài đặt PPO

Mất mát chính sách là âm của trung bình giá trị nhỏ hơn giữa `ratio * A` và `clamp(ratio, 1-eps, 1+eps) * A`. Việc lấy giá trị nhỏ hơn áp dụng cho cả lợi thế dương lẫn âm; nhờ đó PPO hạn chế cả tăng xác suất quá mức với hành động tốt và giảm xác suất quá mức với hành động xấu.

Critic dùng sai số bình phương trung bình, không nhân 1/2. Khi bật `clip_vloss`, lấy giá trị lớn hơn giữa sai số thông thường và sai số của giá trị đã cắt quanh dự đoán cũ. Mất mát tổng là `policy_loss + vf_coef * value_loss - ent_coef * entropy`.

Lợi thế được chuẩn hóa riêng trên mỗi minibatch bằng độ lệch chuẩn tổng thể (`unbiased=False`), cộng `1e-8` vào mẫu số. Returns giữ nguyên. Cách này cũng tránh NaN khi minibatch chỉ có một mẫu. Các đại lượng thuộc lần thu dữ liệu cũ được giữ cố định, gradient chỉ đi qua đầu ra hiện tại của mạng.

Vòng cập nhật xáo dữ liệu ở mỗi epoch, chia thành các minibatch, tính loss, xóa gradient, lan truyền ngược, cắt chuẩn gradient rồi cập nhật optimizer. Ngưỡng cắt mặc định là 0.5. Nếu số mẫu không chia hết, các minibatch có thể lệch nhau một mẫu và vẫn dùng hết dữ liệu. Dừng sớm sau một epoch nếu KL trung bình vượt `target_kl`. Hệ số entropy được đọc từ cấu hình mỗi lần gọi, phục vụ lịch giảm entropy trong vòng huấn luyện.

Các chỉ số theo dõi gồm `approx_kl = mean((ratio - 1) - log_ratio)` và `clipfrac = mean(abs(ratio - 1) > eps)`. Hàm trả trung bình policy loss, value loss, entropy, KL, clipfrac trên các minibatch thực sự đã cập nhật, kèm số bước gradient.

## Bảng kiểm thử hiện tại

Ngày kiểm tra: 26/11/2026. Commit PPO: `8ac89d8`. Môi trường kiểm tra riêng: Python 3.13.7, PyTorch 2.14.0+cpu, pytest 9.2.0.

| Nhóm kiểm tra | Kết quả | Ghi chú |
|---|---|---|
| `tests/test_ppo.py` | 10/10 qua | Gồm ví dụ tính tay, gradient, chuẩn hóa, cập nhật và dừng sớm |
| `tests/test_ppo_update.py` | 9/9 qua | Các trường hợp biên và kiểm tra vòng cập nhật bổ sung |
| `tests/test_model.py` | 0/5 qua | Mạng vẫn là TODO |
| `tests/test_gae.py` | 0/8 qua | GAE vẫn là TODO |
| `tests/test_wrappers.py` | Chưa chạy được | Thiếu `gymnasium` khi thu thập kiểm tra |
| `tests/test_pipeline.py` | Chưa chạy được | Thiếu `omegaconf` khi thu thập kiểm tra |

Bảng này là trạng thái trong lúc làm riêng phần PPO, chưa phải kết quả tích hợp để nộp. Cần chạy lại toàn bộ kiểm thử bằng Python 3.10–3.12 với đủ thư viện sau khi ghép các phần của nhóm, rồi thay bảng bằng số liệu cuối. Chưa có lỗi từ một lần huấn luyện thực để phân tích. Phần GAE sẽ được đọc chéo khi TV3 hoàn thành.

## Thí nghiệm E2: entropy cố định

Giả thuyết: giữ hệ số entropy 0.01 duy trì khám phá lâu hơn, đổi lại có thể làm chậm việc ổn định chính sách so với lịch giảm entropy của E0. E2 dự kiến chạy 2 triệu bước trên 1-1, seed 1, đối chiếu cùng đoạn bước đầu của E0. Các lựa chọn về phần thưởng, mạng, môi trường và lịch lr cần giữ giống nhau.

Chưa chạy E2 vì còn thiếu phần tích hợp và cấu hình E0 đã chốt. Đặc biệt, lịch lr hiện phụ thuộc tổng ngân sách: chạy E0 3–5 triệu bước và E2 2 triệu bước sẽ làm lr khác nhau ngay trong đoạn so sánh nếu dùng nguyên vòng huấn luyện. Cần thống nhất điểm này trước khi chạy. Kết quả, hình vẽ và kết luận sẽ bổ sung từ log thực tế.
