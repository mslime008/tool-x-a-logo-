# Xóa logo video — Windows MVP (Phase 1)

Ứng dụng desktop tiếng Việt để inpaint logo/đồ họa **do người dùng chọn** trên video người dùng sở hữu hoặc được phép chỉnh sửa. Chạy CPU, không cần CUDA/RTX; GeForce MX330 2 GB không phải điều kiện chạy. Đây là MVP có kiểm thử, chưa phải bản Windows đóng gói/phát hành đã qua QA trên máy thật.

## Cài và chạy trên Windows 11

1. Cài **Python 3.11 64 bit** từ <https://www.python.org/downloads/windows/> (chọn Python Launcher và thêm PATH).
2. Cài **FFmpeg 7.1 hoặc mới hơn**, bản có `libx264` và `libx265`. Trang chính thức <https://ffmpeg.org/download.html#build-windows> dẫn tới các nhà cung cấp build Windows. Giải nén, thêm thư mục `bin` chứa **cả ffmpeg.exe và ffprobe.exe** vào PATH rồi mở PowerShell mới.
3. Mở PowerShell trong thư mục repository:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
ffmpeg -version
ffprobe -version
ffmpeg -hide_banner -encoders | Select-String "libx264|libx265"
.\.venv\Scripts\python.exe -m logo_remover.app
```

Các lần sau: nhấp đúp **run_windows.bat**. Không cần kích hoạt PowerShell script, API key hoặc tài khoản cloud. Cài các gói bằng HTTPS mặc định, không tắt kiểm tra chứng chỉ. `opencv-python-headless` cung cấp xử lý ảnh; toàn bộ GUI dùng PySide6 nên không cần `cv2.imshow`.

## Quy trình sử dụng

1. **Nhập video** MP4/MOV hoặc kéo thả một video vào cửa sổ. Chỉ thêm nhiều video lần lượt sau khi đã đưa từng video vào hàng đợi.
2. Dùng thanh thời gian hoặc **Phát** để xem nội dung. Preview chạy bằng worker, lấy mẫu khoảng 10 fps, **không phát âm thanh**; file xuất giữ âm thanh. Với video xoay, trình sửa hiển thị hướng pixel mã hóa gốc và báo góc xoay; file xuất giữ display rotation. Tỉ lệ pixel không vuông được phản ánh trong preview.
3. Vẽ **Hình chữ nhật**, **Đa giác** (nhấp đúp để kết thúc), hoặc **Cọ vẽ**. Chuột phải bỏ nét đang vẽ. Chỉ vẽ trên chế độ **Trước xử lý**. Nới mask để bao trọn mép logo; mềm biên được giới hạn **vào trong** vùng mask đã nới.
4. Mỗi mask có thời gian bắt đầu/kết thúc riêng: khoảng **[bắt đầu, kết thúc)**, tính bằng giây tương đối từ frame đầu tiên. Chọn mask, sửa thông số, bấm **Áp dụng cho mask đã chọn**. Trong Phase 1, mask đứng yên theo tọa độ; chưa có tracking.
5. Chọn TELEA hoặc Navier–Stokes, bán kính; bấm **Tính xem trước** tại mốc hiện tại. Kiểm tra **Sau xử lý**, **Song song**, **Sai khác ×4** và **Phóng to chi tiết** (100/200/400%, có cuộn). Preview so sánh frame trước mã hóa, không thay thế kiểm tra file xuất.
6. Chọn thư mục đầu ra, H.264/H.265, CRF (mặc định 18), tốc độ mã hóa. CRF thấp cho ít mất chi tiết hơn, file lớn hơn; không có cam kết phục hồi lossless.
7. Bấm **Thêm tác vụ và xuất**. Mỗi tác vụ giữ bản sao riêng của mask/thông số, nên có thể chuẩn bị video tiếp theo trong khi xuất. Tác vụ chạy **lần lượt**, có phần trăm, chi tiết lỗi, hủy và thử lại. Tiến độ 0–75% là xử lý frame, 78% là mã hóa, 93–100% là ghép/publish; mã hóa có thể giữ cùng một phần trăm khá lâu.
8. **Nạp thiết lập tác vụ để chỉnh sửa** phục hồi mask/thông số vào trình sửa. Sau khi sửa, thêm **tác vụ mới**; bản đã xếp hàng không bị thay đổi. Hủy bản cũ nếu không cần. Hàng đợi chưa lưu qua lần đóng ứng dụng.

Đầu ra có hậu tố ngẫu nhiên, không ghi đè file đã có hoặc nguồn, kể cả khi hai tác vụ chọn cùng đích. Kết quả chỉ xuất hiện khi hoàn tất. Hard-link atomic publication yêu cầu filesystem hỗ trợ hard links (NTFS trên Windows; ổ FAT/exFAT hoặc một số ổ mạng sẽ báo lỗi thay vì ghi đè).

## Chất lượng, thời gian và bộ nhớ

- Local là **OpenCV inpainting thật**, không blur/crop/che đen. Tái tạo theo ngữ cảnh lân cận; logo lớn, nền phức tạp/chuyển động có thể cho kết quả kém và nhấp nháy. Không có phục hồi hoàn hảo hoặc bảo đảm giữ 100% chất lượng.
- Frame không có mask hoạt động được trả nguyên từ bộ xử lý; pixel BGR 8 bit **ngoài vùng mask đã nới** được giữ bit-for-bit trước mã hóa. Giải mã YUV sang BGR và mã hóa lại H.264/H.265 vẫn có chuyển đổi màu/lấy mẫu chroma và mất chi tiết trên toàn frame. CRF 0 cũng không khôi phục thông tin đã mất ở bước BGR/YUV420.
- Không resize: giữ kích thước nguồn và SAR; xuất yuv420p 8 bit để tương thích rộng. Kích thước lẻ bị chặn thay vì âm thầm crop/pad. Video >8 bit SDR có cảnh báo giảm độ sâu màu. **HDR PQ/HLG bị chặn xuất** vì pipeline này chưa bảo toàn HDR. Dolby Vision, HDR gain maps, ICC, dynamic HDR metadata và các dạng HDR không gắn transfer chuẩn chưa được hỗ trợ/QA.
- FFprobe lấy PTS từng frame; decode `-noautorotate`/passthrough; PNG lossless từng frame trên đĩa; concat theo thời gian microsecond; encode VFR bằng CPU. Giữ số frame và mốc trình chiếu, không ép video VFR thành CFR. B-frame encode tắt để hiệu chỉnh duration gói cuối ổn định. Nguồn thiếu/PTS không tăng bị từ chối. Timestamp đầu được chuẩn hóa về 0; offset audio được dịch tương ứng. Giới hạn chính xác timeline là microsecond và timebase audio/container.
- Giữ display rotation, SAR, color range/matrix/primaries/transfer đã biết qua FFmpeg khi hỗ trợ. Pipeline dùng BGR 8 bit nên không bảo toàn chính xác mọi màu; **không tự gắn BT.709** cho nguồn thiếu metadata. Chưa hỗ trợ display matrix có flip/skew, edit list phức tạp, interlaced video hoặc nhiều luồng video; dùng luồng video đầu tiên.
- Âm thanh AAC/MP3/ALAC/AC3/EAC3 được stream-copy vào MP4. Đầu vào codec khác chuyển AAC 192 kb/s (hiện ở trạng thái tác vụ). Với API xuất MOV có thể stream-copy codec MOV hỗ trợ; nếu FFmpeg không hỗ trợ tổ hợp đó sẽ báo lỗi. GUI MVP xuất MP4. Không dùng `-shortest` để tránh cắt mất đuôi audio. Có thể giữ nhiều luồng audio; heuristic tương thích dựa trên luồng đầu nên file codec trộn có thể báo lỗi. Subtitles, chapters, data tracks và attachments chưa được giữ.
- RAM không tăng theo tổng số frame: decode/xử lý từng frame, encoder 2 thread, 1 export tại một thời điểm. 4K vẫn cần RAM cho một số buffer/frame và codec (nên có **8 GB RAM trở lên**, chưa đo peak RAM trên MX330/Windows). Mốc PTS nhỏ được lưu trong RAM. Tệp tạm có thể rất lớn: 4K 15 giây × 30 fps cần khoảng **10.5 GiB + phần dự phòng** ở trường hợp PNG khó nén. Kiểm tra dung lượng trước export; xử lý có thể chậm trên CPU laptop. Không tải cả video vào VRAM/RAM.
- Tệp tạm nằm trong thư mục riêng `logo-remover-*` tại thư mục đầu ra, được xóa khi thành công/lỗi/hủy có kiểm soát. Nếu hệ điều hành/app bị kill hoặc mất điện, thư mục có thể còn; chỉ xóa các thư mục tạm sau khi bảo đảm không có tác vụ chạy. Mặc định không lưu raw media hay secrets trong Git. Xóa file không phải secure erase trên SSD.

## Kiến trúc

```text
src/logo_remover/
  app.py          GUI, preview/background tasks, vẽ mask, so sánh
  metadata.py     ffprobe, metadata và PTS
  timeline.py     kích hoạt mask theo thời gian
  masks.py        rectangle/polygon/brush, padding và feather
  inpainting.py   TELEA / Navier–Stokes, inpaint ROI
  export.py       FFmpeg decode, tệp tạm, VFR encode, audio/metadata
  queue.py        snapshot mỗi tác vụ, worker và cancellation
  settings.py     QSettings, chỉ lưu thư mục đầu ra không chứa secrets
  providers.py    interface cloud và mock rõ nhãn
schemas/cloud-job-v1.json
```

Không gọi shell khi xử lý đường dẫn media; subprocess dùng danh sách argument. FFmpeg/ffprobe phải được cài từ nguồn tin cậy; ứng dụng chưa sandbox parser media, chỉ dùng media tin cậy. Tôn trọng giấy phép FFmpeg/x264/x265 khi đóng gói/phân phối; packaging nằm ngoài Phase 1.

## Cloud: chỉ hợp đồng mở rộng, chưa có AI backend

UI **Đám mây AI chất lượng** báo chưa tích hợp và không tải dữ liệu lên. **Tự động** dùng local; không có fallback upload âm thầm. `CloudProvider` có `submit`, `poll`, `cancel`, `retrieve`, `delete`; không endpoint/provider/model thực nào được gán sẵn. `MockProvider` chỉ mô phỏng trạng thái trong RAM, nhãn `MOCK`, không đọc/upload video và `retrieve` luôn báo không có kết quả AI.

Job JSON v1.0 gồm metadata (không chứa local path), mask/tọa độ/thời gian, tên model, thiết lập xuất và xác nhận upload; schema tại `schemas/cloud-job-v1.json`. Adapter thực ở Phase 3 **phải**:

- Xác thực schema và quan hệ `start < end`, giới hạn kích thước và thời lượng, capability/model thực; không chỉ tin JSON do client gửi.
- Xin xác nhận rõ ràng cho **từng job** với tên provider, tên video, phạm vi upload, vùng dữ liệu, model, thời gian lưu và chính sách xóa; timestamp consent chỉ có hiệu lực cho lần submit đó. Không chọn hộ hoặc tự submit khi ở Auto.
- Dùng TLS có xác minh, upload URL ngắn hạn, object ID ngẫu nhiên, storage riêng tư và encryption at rest; không gửi local path hoặc key trong JSON/log.
- Hạn lưu tối đa 24 giờ, xóa input/mask/result sau tải kết quả hoặc sau hủy; cung cấp trạng thái/xác nhận xóa và dọn bản bỏ dở. Retention phía server phải thực thi dù client mất kết nối. Không tuyên bố mock đang thực thi TTL/storage vì mock không lưu media.
- Giữ key ngoài Git bằng biến môi trường dành riêng adapter hoặc Windows Credential Manager; key không nằm trong QSettings/job JSON. Không có API key cần thiết trong Phase 1. Không hardcode endpoint.

## Kiểm thử

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Trên Linux/headless:

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q
```

Bộ test gồm rasterization/padding/feather và tính bất biến ngoài mask, bật/tắt theo thời gian, metadata HDR/rotation/fps, snapshot queue độc lập, lifecycle/schema mock, Qt khởi tạo/nhập media/vẽ mask/xuất qua queue headless. Integration tạo video **8 giây** có nền chuyển động/logo/âm thanh, xuất cả H.264/H.265 và kiểm tra frame count/PTS/màu/audio hash/nguồn không đổi; thêm VFR/SAR/rotation, hủy/dọn tạm và một frame cho từng độ phân giải 1080p/4K. Không có GPU. GitHub Actions chạy Linux với FFmpeg và Windows headless unit/Qt smoke; Windows full FFmpeg integration cần chạy trên máy đích.

**Kiểm thử offscreen không phải kiểm thử giao diện trực quan trên Windows.** Trước phát hành: chạy trên Windows 11/MX330, thử drag/drop, polygon/brush khi preview letterbox, tiếng Việt, playback/scrub liên tục, cảnh báo HDR, clip thực 1080p/4K 8–15s, màu và đồng bộ tiếng/hình, cancel giữa encode, disk đầy, đường dẫn Unicode, file khóa, đóng app. Đo RAM/thời gian và kiểm tra thành phẩm bằng player độc lập.

## Các giai đoạn sau (chưa triển khai)

Phase 2: tracking logo di chuyển, cải thiện độ ổn định theo thời gian và điều khiển mask nâng cao. Phase 3: adapter GPU độc lập có consent/storage thật. Phase 4: batch nâng cao, QA, installer/Windows packaging. Phase 1 đã có mask theo khoảng thời gian cơ bản và batch queue trong phiên; không có mô hình deep learning cục bộ hoặc cloud thực.
