# Ghi chú TV4

## PPO — 26/11/2026

- Nhánh: `tv4/ppo`.
- Mã ban đầu: `acc5af4`; commit cài đặt PPO: `8ac89d8`.
- Đã làm: `clipped_policy_loss`, `value_loss`, `approx_kl_and_clipfrac`, `ppo_loss`, `ppo_update`.
- Giữ nguyên `compute_gae` của TV3, `model.py`, wrapper, cấu hình mặc định và mã đánh giá.
- Chuẩn hóa lợi thế theo từng minibatch, dùng `std(unbiased=False)`; không chuẩn hóa returns. Dữ liệu của chính sách cũ và mục tiêu được tách khỏi gradient.
- Mỗi epoch xáo mẫu một lần, chia đúng số minibatch, giữ cả mẫu dư nếu số mẫu không chia hết. Cắt gradient theo `cfg.max_grad_norm` (mặc định 0.5).
- Xét dừng sớm sau khi hoàn thành epoch, dùng trung bình KL của epoch đó. Thống kê trả về là trung bình của tất cả minibatch đã chạy.
- `ent_coef` được đọc mỗi lần tính loss; TV2 có thể thay đổi hệ số này theo bước huấn luyện.

Kiểm tra trên Python 3.13.7, PyTorch 2.14.0+cpu, pytest 9.2.0 trong môi trường riêng `D:\Mario-RL\.venv-test`:

```powershell
D:\Mario-RL\.venv-test\Scripts\python.exe -m pytest -q tests/test_ppo.py tests/test_ppo_update.py
```

Kết quả: **19 passed** (10 bài có sẵn, 9 bài bổ sung). Các bài bổ sung kiểm tra minibatch một mẫu, không đổi dữ liệu đầu vào, không bỏ mẫu dư, xáo lại giữa các epoch, cắt gradient trước bước optimizer, dừng theo KL trung bình epoch, thống kê trung bình và thay đổi hệ số entropy.

Kiểm tra riêng mạng và GAE: 13 failed, đều do `NotImplementedError` (5 mạng, 8 GAE). Chạy toàn bộ `pytest -q` bị dừng ở bước thu thập: thiếu `omegaconf` cho pipeline, thiếu `gymnasium` cho wrapper. Chưa có kết quả kiểm thử toàn hệ thống. README yêu cầu Python 3.10–3.12 cho môi trường trò chơi; kết quả trên Python 3.13 chỉ xác nhận phần tính toán PPO độc lập.

TV3 cần duyệt phần PPO trước khi gộp. Phần duyệt chéo `compute_gae` chờ TV3 cài đặt xong.

## E2 — chờ tích hợp và chốt E0

Chưa chạy huấn luyện, chưa có số bước, sps, checkpoint hoặc flag_rate để báo cáo.

E2 kiểm tra tác động của lịch entropy. Giả thuyết: giữ `ent_coef=0.01` giúp duy trì khám phá nhưng có thể làm chính sách hội tụ chậm hơn lịch giảm của E0. So sánh với 2 triệu bước đầu của E0, cùng seed 1 và giữ nguyên các lựa chọn khác.

Khi TV2 gửi cấu hình E0 đã chốt:

1. Sao cấu hình E0 thành `configs/1-1-tv4-e2.yaml`; không tự chọn lại trọng số, thưởng cờ hay ngưỡng không tiến.
2. Giữ `level: "1-1"`, `seed: 1`, `resume: null`; đặt `algo.ent_coef: 0.01`, `algo.ent_coef_end: null`, `algo.total_steps: 2000000`, `run_dir: runs/1-1/tv4-e2-s1`.
3. Soát các khác biệt: chỉ lịch entropy là yếu tố thí nghiệm; ngân sách chạy và thư mục đầu ra phục vụ việc đối chiếu. E0 có ngân sách dài hơn nên lịch giảm lr theo `total_steps` trong `train.py` hiện tại cũng sẽ khác: cần TV2 thống nhất cách giữ cùng lịch lr trên đoạn so sánh trước khi chạy, tránh biến E2 thành thí nghiệm đổi hai yếu tố.
4. Đợi cả bộ pytest qua, commit cấu hình và mã tích hợp; ghi commit, cấu hình máy và lệnh chạy tại đây rồi mới khởi động.
5. Ghi thời gian, sps, đường dẫn checkpoint và nhận xét từ dữ liệu thực. Đối chiếu return, x_pos, flag_rate, entropy, approx_kl và clipfrac theo số bước; dùng cùng giao thức đánh giá cho cả hai nhánh.

```powershell
python -m mario_ppo.train --config configs/1-1-tv4-e2.yaml
```

Chưa tạo cấu hình E2 để chạy vì chưa có E0 làm đối chứng. `ent_coef_end` hiện cũng chưa được `train.py` hỗ trợ.

## Các việc còn phụ thuộc

| Việc | Cần có trước |
|---|---|
| Chạy thử khoảng 200k bước, đo sps | Wrapper TV1, mạng TV2, GAE TV3, môi trường Python phù hợp |
| E2 | Các phần cốt lõi đã tích hợp, cấu hình E0 và cách giữ cùng lịch lr |
| P3 màn 3-3 | Checkpoint E0 của nhóm và quy ước khởi tạo học chuyển giao từ TV2 |
| B2 seed 2 | Màn thưởng và cấu hình B1 do nhóm chốt |
| Duyệt GAE | Cài đặt `compute_gae` của TV3 |
| Đánh giá cuối, JSON, video | Checkpoint được chọn cho 1-1 và màn thưởng |
| Đóng gói bài nộp | Log, cấu hình, checkpoint, báo cáo PDF và slide của nhóm |

Lưu ý cho P3/B2: `resume` hiện nạp cả optimizer lẫn `global_step`. Nếu checkpoint E0 đã vượt 1 triệu bước, đặt `algo.total_steps=1000000` sẽ không huấn luyện thêm. Cần TV2 thống nhất cách khởi tạo học chuyển giao và cách tính ngân sách; không dùng lệnh resume như một lần chạy mới mà chưa kiểm tra bước bắt đầu.
