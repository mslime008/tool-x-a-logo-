from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import sys
import uuid
import time
from dataclasses import replace
import cv2
import numpy as np
from PySide6.QtCore import Qt, QPointF, QRectF, QRunnable, QThreadPool, QTimer, Signal, QObject, Slot
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QComboBox, QSlider, QSpinBox,
    QDoubleSpinBox, QListWidget, QGroupBox, QFormLayout, QSplitter, QScrollArea, QDialog,
    QGraphicsView, QGraphicsScene)
from .export import ExportSettings, read_frame
from .inpainting import inpaint
from .masks import Mask
from .metadata import probe
from .queue import Job, ExportWorker
from .settings import preferences
from .cloud_client import CloudOptions, HttpGPUProvider


class TaskSignals(QObject):
    result = Signal(object)
    error = Signal(str)


class Task(QRunnable):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.signals = TaskSignals()

    @Slot()
    def run(self):
        try:
            self.signals.result.emit(self.fn())
        except Exception as exc:
            self.signals.error.emit(str(exc))


def qimage(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888).copy()


class Canvas(QWidget):
    drawn = Signal(str, object)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(440, 280)
        self.frame = None
        self.image = None
        self.masks = []
        self.seconds = 0.0
        self.sar = 1.0
        self.tool = "rectangle"
        self.points = []
        self.dragging = False
        self.show_masks = True
        self.setMouseTracking(True)

    def set_frame(self, frame):
        self.frame = frame
        self.image = qimage(frame)
        self.update()

    def image_rect(self):
        if self.image is None:
            return QRectF()
        w, h = self.image.width() * self.sar, self.image.height()
        scale = min(self.width() / w, self.height() / h)
        return QRectF((self.width() - w * scale) / 2, (self.height() - h * scale) / 2,
                      w * scale, h * scale)

    def media_point(self, point):
        rect = self.image_rect()
        if self.image is None or not rect.contains(point):
            return None
        return (max(0, min(self.image.width() - 1, (point.x() - rect.x()) * self.image.width() / rect.width())),
                max(0, min(self.image.height() - 1, (point.y() - rect.y()) * self.image.height() / rect.height())))

    def display_point(self, point):
        r = self.image_rect()
        return QPointF(r.x() + point[0] * r.width() / self.image.width(),
                       r.y() + point[1] * r.height() / self.image.height())

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#10141c"))
        if self.image is None:
            painter.setPen(QColor("#b5c0d5"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Nhập hoặc kéo thả video MP4 / MOV")
            return
        painter.drawImage(self.image_rect(), self.image)
        if self.show_masks:
            for mask in self.masks:
                color = "#37e2b4" if mask.start <= self.seconds < mask.end else "#666c79"
                self.paint_shape(painter, mask.kind, mask.points, color, mask.brush_size)
            if self.points:
                self.paint_shape(painter, self.tool, self.points, "#ffd166", 20)

    def paint_shape(self, painter, kind, points, color, brush_size):
        if not points:
            return
        painter.setPen(QPen(QColor(color), 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        ps = [self.display_point(p) for p in points]
        if kind == "rectangle" and len(ps) >= 2:
            painter.drawRect(QRectF(ps[0], ps[-1]).normalized())
        elif kind == "polygon":
            painter.drawPolygon(QPolygonF(ps))
        else:
            if kind == "brush":
                painter.setPen(QPen(QColor(color), max(2, brush_size * self.image_rect().width() / self.image.width()),
                                    Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            if len(ps) == 1:
                painter.drawPoint(ps[0])
            else:
                painter.drawPolyline(QPolygonF(ps))

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            self.points = []
            self.update()
            return
        point = self.media_point(event.position())
        if point is None or event.button() != Qt.MouseButton.LeftButton or not self.show_masks:
            return
        if self.tool == "polygon":
            self.points.append(point)
        else:
            self.points = [point, point] if self.tool == "rectangle" else [point]
            self.dragging = True
        self.update()

    def mouseMoveEvent(self, event):
        point = self.media_point(event.position())
        if point and self.dragging:
            if self.tool == "rectangle":
                self.points[-1] = point
            else:
                self.points.append(point)
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.dragging:
            self.dragging = False
            self.finish_shape()

    def mouseDoubleClickEvent(self, event):
        if self.tool == "polygon" and len(self.points) >= 3:
            self.finish_shape()

    def finish_shape(self):
        if self.points:
            self.drawn.emit(self.tool, list(self.points))
        self.points = []
        self.update()


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Xóa logo video • MVP cục bộ")
        self.resize(1280, 900)
        self.setAcceptDrops(True)
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(3)
        self.export_pool = QThreadPool(self)
        self.export_pool.setMaxThreadCount(1)
        self.meta = None
        self.masks = []
        self.original = None
        self.repaired = None
        self.jobs = []
        self.active_job = None
        self.worker = None
        self.tasks = {}
        self.request_id = 0
        self.preview_busy = False
        self.pending_preview = None
        self.loading = False
        self.closing = False
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.tick)
        self.build_ui()
        self.setStyleSheet('''
            QWidget { background:#181e2a; color:#e1e7f2; font-size:13px; }
            QPushButton { background:#2a3850; padding:8px; border-radius:5px; }
            QPushButton:hover { background:#365578; }
            QPushButton:disabled { color:#687487; }
            QComboBox,QSpinBox,QDoubleSpinBox,QListWidget { background:#111722; padding:4px; }
            QGroupBox { border:1px solid #3b475b; border-radius:6px; margin-top:12px; padding-top:14px; }
            QGroupBox::title { subcontrol-origin:margin; left:10px; }
        ''')

    def build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        toolbar = QHBoxLayout()
        self.import_button = QPushButton("Nhập video")
        self.import_button.clicked.connect(self.choose_input)
        toolbar.addWidget(self.import_button)
        self.info = QLabel("Sẵn sàng • CPU • Không tự tải lên đám mây")
        self.info.setWordWrap(True)
        toolbar.addWidget(self.info, 1)
        layout.addLayout(toolbar)
        splitter = QSplitter()
        layout.addWidget(splitter, 1)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.canvas = Canvas()
        self.canvas.drawn.connect(self.add_mask)
        left_layout.addWidget(self.canvas, 1)
        bar = QHBoxLayout()
        self.play = QPushButton("Phát")
        self.play.clicked.connect(self.toggle_play)
        bar.addWidget(self.play)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self.seek)
        bar.addWidget(self.slider, 1)
        self.clock = QLabel("0.000 s")
        bar.addWidget(self.clock)
        left_layout.addLayout(bar)
        quality = QHBoxLayout()
        self.view_mode = QComboBox()
        self.view_mode.addItems(["Trước xử lý", "Sau xử lý", "Song song", "Sai khác ×4"])
        self.view_mode.currentIndexChanged.connect(self.show_comparison)
        quality.addWidget(self.view_mode)
        self.preview_button = QPushButton("Tính xem trước")
        self.preview_button.clicked.connect(self.compute_preview)
        quality.addWidget(self.preview_button)
        detail = QPushButton("Phóng to chi tiết")
        detail.clicked.connect(self.magnify)
        quality.addWidget(detail)
        left_layout.addLayout(quality)
        note = QLabel("Vẽ trên chế độ Trước xử lý • Đa giác: nhấp đúp để kết thúc • Chuột phải: bỏ nét đang vẽ\n"
                      "Inpainting từng khung có thể nhấp nháy hoặc lỗi trên nền chuyển động; không phục hồi hoàn hảo.")
        note.setWordWrap(True)
        left_layout.addWidget(note)
        splitter.addWidget(left)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        mask_box = QGroupBox("Mask độc lập theo thời gian")
        form = QFormLayout(mask_box)
        self.tool = QComboBox()
        self.tool.addItems(["Hình chữ nhật", "Đa giác", "Cọ vẽ"])
        self.tool.currentIndexChanged.connect(self.change_tool)
        form.addRow("Công cụ", self.tool)
        self.mask_list = QListWidget()
        self.mask_list.setMaximumHeight(110)
        self.mask_list.currentRowChanged.connect(self.select_mask)
        form.addRow(self.mask_list)
        self.start = QDoubleSpinBox()
        self.end = QDoubleSpinBox()
        for spin in (self.start, self.end):
            spin.setDecimals(3)
            spin.setRange(0, 100000)
            spin.setSuffix(" s")
        self.end.setValue(99999)
        self.padding = QSpinBox()
        self.padding.setRange(0, 100)
        self.padding.setValue(3)
        self.feather = QSpinBox()
        self.feather.setRange(0, 50)
        self.feather.setValue(2)
        self.brush = QSpinBox()
        self.brush.setRange(1, 500)
        self.brush.setValue(20)
        for label, widget in [("Bắt đầu", self.start), ("Kết thúc", self.end), ("Nới mask (px)", self.padding),
                              ("Mềm biên vào trong (px)", self.feather), ("Đường kính cọ (px)", self.brush)]:
            form.addRow(label, widget)
        apply_button = QPushButton("Áp dụng cho mask đã chọn")
        apply_button.clicked.connect(self.apply_mask)
        form.addRow(apply_button)
        remove = QPushButton("Xóa mask đã chọn")
        remove.clicked.connect(self.remove_mask)
        form.addRow(remove)
        right_layout.addWidget(mask_box)
        export_box = QGroupBox("Xuất và hàng đợi")
        form = QFormLayout(export_box)
        self.mode = QComboBox()
        self.mode.addItems(["Cục bộ nhanh", "Đám mây GPU (cần cấu hình)", "Tự động (cục bộ)"])
        form.addRow("Chế độ", self.mode)
        self.mode.currentIndexChanged.connect(lambda index: self.statusBar().showMessage(
            "Xem trước dùng thuật toán cục bộ; model GPU chỉ được dùng cho tác vụ cloud đã xác nhận."
            if index == 1 else "Cục bộ / Tự động không tải video lên cloud."))
        self.cloud_retention = QSpinBox()
        self.cloud_retention.setRange(1, 24)
        self.cloud_retention.setValue(1)
        self.cloud_retention.setSuffix(" giờ")
        form.addRow("Giữ tệp cloud tối đa", self.cloud_retention)
        self.method = QComboBox()
        self.method.addItems(["TELEA", "Navier–Stokes"])
        form.addRow("Thuật toán", self.method)
        self.radius = QSpinBox()
        self.radius.setRange(1, 20)
        self.radius.setValue(3)
        form.addRow("Bán kính (px)", self.radius)
        self.codec = QComboBox()
        self.codec.addItems(["H.264", "H.265"])
        form.addRow("Codec CPU", self.codec)
        self.crf = QSpinBox()
        self.crf.setRange(0, 51)
        self.crf.setValue(18)
        form.addRow("CRF (thấp = ít mất chi tiết)", self.crf)
        self.preset = QComboBox()
        self.preset.addItems(["Nhanh", "Cân bằng", "Chậm"])
        self.preset.setCurrentIndex(1)
        form.addRow("Tốc độ mã hóa", self.preset)
        self.output_dir = str(preferences().value("output_dir", str(Path.home() / "Videos" / "LogoRemover")))
        choose = QPushButton("Chọn thư mục đầu ra")
        choose.clicked.connect(self.choose_output)
        form.addRow(choose)
        self.output_label = QLabel(self.output_dir)
        self.output_label.setWordWrap(True)
        form.addRow(self.output_label)
        enqueue = QPushButton("Thêm tác vụ và xuất")
        enqueue.clicked.connect(self.enqueue)
        form.addRow(enqueue)
        self.queue_list = QListWidget()
        self.queue_list.setMinimumHeight(100)
        self.queue_list.currentRowChanged.connect(self.queue_detail)
        form.addRow(self.queue_list)
        actions = QHBoxLayout()
        cancel = QPushButton("Hủy đã chọn")
        cancel.clicked.connect(self.cancel_job)
        retry = QPushButton("Thử lại")
        retry.clicked.connect(self.retry_job)
        actions.addWidget(cancel)
        actions.addWidget(retry)
        form.addRow(actions)
        load = QPushButton("Nạp thiết lập tác vụ để chỉnh sửa")
        load.clicked.connect(self.load_job)
        form.addRow(load)
        self.job_info = QLabel("Mỗi tác vụ lưu riêng mask và thiết lập xuất.")
        self.job_info.setWordWrap(True)
        form.addRow(self.job_info)
        right_layout.addWidget(export_box)
        scroll.setWidget(right)
        splitter.addWidget(scroll)
        splitter.setSizes([830, 410])
        self.statusBar().showMessage("Cần FFmpeg/ffprobe trong PATH; không cần GPU.")

    def task(self, fn, success, failure=None):
        task = Task(fn)
        token = uuid.uuid4().hex
        self.tasks[token] = task
        def finish(value):
            self.tasks.pop(token, None)
            success(value)
        def failed(text):
            self.tasks.pop(token, None)
            (failure or self.error)(text)
        task.signals.result.connect(finish)
        task.signals.error.connect(failed)
        self.pool.start(task)

    def error(self, text):
        QMessageBox.warning(self, "Thông báo", str(text))

    def choose_input(self):
        path, _ = QFileDialog.getOpenFileName(self, "Nhập video", "", "Video (*.mp4 *.mov)")
        if path:
            self.open_video(path)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        if paths:
            self.open_video(paths[0])
            if len(paths) > 1:
                self.statusBar().showMessage("MVP nhập từng video; thêm vào hàng đợi trước khi nhập video tiếp theo.")

    def open_video(self, path, restored=None):
        if self.loading:
            return
        self.stop_play()
        self.loading = True
        self.import_button.setEnabled(False)
        self.request_id += 1
        self.pending_preview = None
        def read():
            meta = probe(path)
            return meta, read_frame(meta, 0)
        def complete(result):
            self.loading = False
            self.import_button.setEnabled(True)
            self.meta, self.original = result
            self.masks = deepcopy(restored.masks) if restored else []
            self.canvas.masks = self.masks
            self.canvas.points = []
            try:
                n, d = self.meta.sar.split(":")
                self.canvas.sar = float(n) / float(d) if float(d) and float(n) else 1
            except (ValueError, ZeroDivisionError):
                self.canvas.sar = 1
            self.repaired = None
            self.slider.blockSignals(True)
            self.slider.setRange(0, max(0, int((self.meta.duration - 1 / self.meta.rate) * 1000)))
            self.slider.setValue(0)
            self.slider.blockSignals(False)
            self.canvas.seconds = 0
            self.clock.setText("0.000 s")
            self.start.setValue(0)
            self.end.setValue(self.meta.duration)
            if restored:
                self.method.setCurrentIndex(0 if restored.settings.method == "TELEA" else 1)
                self.radius.setValue(restored.settings.radius)
                self.codec.setCurrentIndex(0 if restored.settings.codec == "h264" else 1)
                self.crf.setValue(restored.settings.crf)
                self.preset.setCurrentIndex(["fast", "medium", "slow"].index(restored.settings.preset))
            self.refresh_masks()
            self.view_mode.setCurrentIndex(0)
            self.show_comparison()
            meta = self.meta
            self.info.setText(f"{Path(meta.path).name} • {meta.width}×{meta.height} • {meta.duration:.2f} s • "
                              f"{meta.rate:.3f} fps trung bình • Xoay {meta.rotation}°")
            warnings = []
            if meta.hdr:
                warnings.append("Nguồn HDR: xem trước SDR 8 bit không chính xác; xuất bị chặn để tránh mất màu HDR.")
            elif any(x in meta.pix_fmt for x in ("10", "12", "16")):
                warnings.append("Nguồn sâu màu trên 8 bit sẽ giảm về SDR 8 bit trong pipeline local.")
            if meta.rotation:
                warnings.append("Để mask khớp pixel nguồn, trình sửa hiển thị hướng mã hóa gốc. Metadata xoay được giữ khi xuất.")
            if warnings:
                self.error("\n".join(warnings))
        def failed(text):
            self.loading = False
            self.import_button.setEnabled(True)
            self.error(text)
        self.task(read, complete, failed)

    def change_tool(self, index):
        self.canvas.tool = ["rectangle", "polygon", "brush"][index]
        self.canvas.points = []
        self.canvas.update()

    def new_mask(self, kind, points):
        mask = Mask(kind, points, self.start.value(), self.end.value(), self.padding.value(),
                    self.feather.value(), self.brush.value())
        mask.validate()
        return mask

    def add_mask(self, kind, points):
        if self.meta is None:
            return
        try:
            self.request_id += 1
            self.masks.append(self.new_mask(kind, points))
            self.repaired = None
            self.refresh_masks()
            self.mask_list.setCurrentRow(len(self.masks) - 1)
        except ValueError as exc:
            self.error(str(exc))

    def refresh_masks(self):
        row = self.mask_list.currentRow()
        self.mask_list.blockSignals(True)
        self.mask_list.clear()
        names = {"rectangle": "Chữ nhật", "polygon": "Đa giác", "brush": "Cọ"}
        for index, mask in enumerate(self.masks):
            self.mask_list.addItem(f"{index + 1}. {names[mask.kind]} • {mask.start:.3f}–{mask.end:.3f} s")
        self.mask_list.setCurrentRow(min(row, len(self.masks) - 1))
        self.mask_list.blockSignals(False)
        self.canvas.update()

    def select_mask(self, index):
        if 0 <= index < len(self.masks):
            mask = self.masks[index]
            self.start.setValue(mask.start)
            self.end.setValue(mask.end)
            self.padding.setValue(mask.padding)
            self.feather.setValue(mask.feather)
            self.brush.setValue(mask.brush_size)

    def apply_mask(self):
        index = self.mask_list.currentRow()
        if 0 <= index < len(self.masks):
            old = self.masks[index]
            try:
                self.request_id += 1
                self.masks[index] = self.new_mask(old.kind, old.points)
                self.repaired = None
                self.refresh_masks()
                self.show_comparison()
            except ValueError as exc:
                self.error(str(exc))

    def remove_mask(self):
        index = self.mask_list.currentRow()
        if 0 <= index < len(self.masks):
            self.request_id += 1
            self.masks.pop(index)
            self.repaired = None
            self.refresh_masks()
            self.show_comparison()

    def stop_play(self):
        self.timer.stop()
        self.play.setText("Phát")

    def toggle_play(self):
        if self.meta is None:
            return
        if self.timer.isActive():
            self.stop_play()
        else:
            if self.slider.value() >= self.slider.maximum():
                self.slider.setValue(0)
            self.timer.start()
            self.play.setText("Tạm dừng")

    def tick(self):
        if self.preview_busy or self.loading:
            return
        value = min(self.slider.maximum(), self.slider.value() + 100)
        self.slider.setValue(value)
        if value >= self.slider.maximum():
            self.stop_play()

    def seek(self, value):
        self.clock.setText(f"{value / 1000:.3f} s")
        if self.meta is None or self.loading:
            return
        self.request_id += 1
        self.pending_preview = (self.request_id, self.meta, value / 1000)
        self.launch_preview()

    def launch_preview(self):
        if self.preview_busy or self.pending_preview is None:
            return
        token, meta, seconds = self.pending_preview
        self.pending_preview = None
        self.preview_busy = True
        def complete(frame):
            self.preview_busy = False
            if token == self.request_id:
                self.original, self.repaired = frame, None
                self.canvas.seconds = seconds
                self.show_comparison()
            self.launch_preview()
        def failed(text):
            self.preview_busy = False
            self.stop_play()
            self.statusBar().showMessage(text)
            self.launch_preview()
        self.task(lambda: read_frame(meta, seconds), complete, failed)

    def export_settings(self):
        return ExportSettings("h264" if self.codec.currentIndex() == 0 else "h265", self.crf.value(),
                              ["fast", "medium", "slow"][self.preset.currentIndex()],
                              "TELEA" if self.method.currentIndex() == 0 else "NS", self.radius.value())

    def compute_preview(self):
        if self.original is None or self.loading or self.preview_busy:
            return
        if not self.preview_button.isEnabled():
            return
        self.preview_button.setEnabled(False)
        self.stop_play()
        token = self.request_id
        frame, masks = self.original.copy(), deepcopy(self.masks)
        seconds = self.canvas.seconds
        settings = self.export_settings()
        def complete(result):
            self.preview_button.setEnabled(True)
            if token == self.request_id:
                self.repaired = result
                self.view_mode.setCurrentIndex(1)
                self.show_comparison()
        def failed(text):
            self.preview_button.setEnabled(True)
            self.error(text)
        self.task(lambda: inpaint(frame, masks, seconds, settings.method, settings.radius), complete, failed)

    def comparison(self):
        if self.original is None:
            return None
        mode = self.view_mode.currentIndex()
        if mode == 0 or self.repaired is None:
            return self.original
        if mode == 1:
            return self.repaired
        if mode == 2:
            return np.concatenate((self.original, self.repaired), axis=1)
        return np.clip(cv2.absdiff(self.original, self.repaired).astype(np.uint16) * 4, 0, 255).astype(np.uint8)

    def show_comparison(self, *_):
        self.canvas.show_masks = self.view_mode.currentIndex() == 0
        frame = self.comparison()
        if frame is not None:
            self.canvas.set_frame(frame)
        if self.view_mode.currentIndex() and self.repaired is None:
            self.statusBar().showMessage("Nhấn Tính xem trước để tạo ảnh sau xử lý cho mốc hiện tại.")

    def magnify(self):
        frame = self.comparison()
        if frame is None:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Chi tiết pixel • cuộn để xem vùng ảnh")
        dialog.resize(1000, 700)
        layout = QVBoxLayout(dialog)
        zoom = QComboBox()
        zoom.addItems(["100%", "200%", "400%"])
        layout.addWidget(zoom)
        # Transform a native-resolution pixmap rather than allocate a 4x 4K image.
        from PySide6.QtGui import QPixmap, QTransform
        scene = QGraphicsScene(dialog)
        scene.addPixmap(QPixmap.fromImage(qimage(frame)))
        view = QGraphicsView(scene)
        view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        def resize(index):
            factor = [1, 2, 4][index]
            view.setTransform(QTransform.fromScale(factor, factor))
        zoom.currentIndexChanged.connect(resize)
        layout.addWidget(view)
        dialog.exec()

    def choose_output(self):
        path = QFileDialog.getExistingDirectory(self, "Thư mục đầu ra", self.output_dir)
        if path:
            self.output_dir = path
            preferences().setValue("output_dir", path)
            self.output_label.setText(path)

    def enqueue(self):
        if self.meta is None or self.loading:
            return
        if self.meta.hdr:
            self.error("Không xuất HDR qua pipeline local BGR 8 bit. Hãy chuyển SDR có kiểm soát trước.")
            return
        if not self.masks:
            self.error("Hãy vẽ ít nhất một mask trước khi thêm tác vụ.")
            return
        output = Path(self.output_dir) / (Path(self.meta.path).stem + "_da_xoa_logo_" + uuid.uuid4().hex[:8] + ".mp4")
        job = Job.snapshot(self.meta, self.masks, output, self.export_settings())
        if self.mode.currentIndex() == 1:
            self.prepare_cloud(job)
            return
        self.jobs.append(job)
        self.refresh_jobs()
        self.start_next()

    def confirm_cloud(self, job, options):
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Xác nhận tải lên và xử lý bằng GPU")
        warning = ("\nMÔ HÌNH KIỂM THỬ: chỉ kiểm tra hệ thống, KHÔNG xóa logo." if options.test_model else "")
        dialog.setText(f"Video: {Path(job.meta.path).name}\n"
            f"Nhà cung cấp: {options.provider_name}\nĐịa chỉ: {options.url}\n"
            f"Vùng dữ liệu: {options.data_region}\nChính sách: {options.privacy_policy_url}\n"
            f"Model: {options.model}\nGiữ tệp tối đa: {options.retention_hours} giờ.\n"
            "Video gốc và mask sẽ được tải lên. Yêu cầu xóa sau khi tải kết quả đã xác minh hoặc khi hủy; "
            "hạn lưu vẫn áp dụng nếu mất kết nối. Bạn chịu phí xử lý của nhà cung cấp đã cấu hình. "
            "Chỉ tiếp tục nếu bạn có quyền chỉnh sửa video và chấp thuận chính sách dữ liệu của nhà cung cấp."
            + warning)
        accept = dialog.addButton("Đồng ý tải lên và xử lý", QMessageBox.ButtonRole.AcceptRole)
        dialog.addButton("Không tải lên", QMessageBox.ButtonRole.RejectRole)
        dialog.exec()
        return dialog.clickedButton() is accept

    def prepare_cloud(self, job):
        retention = self.cloud_retention.value()
        def discover():
            provider = HttpGPUProvider.from_environment()
            try:
                return provider.url, provider.capabilities()
            finally:
                provider.close()
        def complete(result):
            if self.closing:
                return
            url, caps = result
            options = CloudOptions(url, caps["provider_name"], caps["model"], time.time(),
                min(retention, caps["retention_hours_max"]), caps["test_model"], uuid.uuid4().hex, caps["capability_revision"],
                caps["data_region"], caps["privacy_policy_url"])
            if not self.confirm_cloud(job, options):
                return
            job.cloud = replace(options, approved_at=time.time())
            self.jobs.append(job)
            self.refresh_jobs()
            self.start_next()
        self.task(discover, complete)

    def refresh_jobs(self):
        row = self.queue_list.currentRow()
        self.queue_list.blockSignals(True)
        self.queue_list.clear()
        for job in self.jobs:
            self.queue_list.addItem(f"{job.id} • {Path(job.meta.path).name} • {job.progress}% • {job.status}")
        self.queue_list.setCurrentRow(row)
        self.queue_list.blockSignals(False)

    def queue_detail(self, row):
        if 0 <= row < len(self.jobs):
            job = self.jobs[row]
            cloud_info = (f"\nCloud: {job.cloud.provider_name} • {job.cloud.model}\nMã máy chủ: {job.remote_id or 'chưa gửi'}"
                          if job.cloud else "")
            self.job_info.setText(f"{job.output}\n{job.settings.codec} CRF {job.settings.crf} • {len(job.masks)} mask{cloud_info}\n{job.error}")

    def start_next(self):
        if self.active_job is not None:
            return
        job = next((j for j in self.jobs if j.status == "Đang chờ"), None)
        if job is None:
            return
        self.active_job = job
        job.status = "Đang xử lý"
        self.worker = ExportWorker(job)
        self.worker.signals.progress.connect(self.job_progress)
        self.worker.signals.finished.connect(self.job_finished)
        self.export_pool.start(self.worker)
        self.refresh_jobs()

    def job_progress(self, job_id, percent, text):
        job = next(j for j in self.jobs if j.id == job_id)
        job.progress = percent
        self.refresh_jobs()
        self.statusBar().showMessage(f"{job_id}: {text} ({percent}%)")

    def job_finished(self, job_id, status, error):
        job = next(j for j in self.jobs if j.id == job_id)
        job.status, job.error = status, error
        self.active_job = None
        self.worker = None
        self.refresh_jobs()
        self.queue_detail(self.queue_list.currentRow())
        if not self.closing:
            self.start_next()

    def selected_job(self):
        row = self.queue_list.currentRow()
        return self.jobs[row] if 0 <= row < len(self.jobs) else None

    def cancel_job(self):
        job = self.selected_job()
        if job and job.status in {"Đang chờ", "Đang xử lý"}:
            job.cancel.set()
            if job.status == "Đang chờ":
                job.status = "Đã hủy"
            self.refresh_jobs()

    def retry_job(self):
        job = self.selected_job()
        if job and job.status in {"Lỗi", "Đã hủy"}:
            if job.cloud:
                if not self.confirm_cloud(job, job.cloud):
                    return
                if job.status == "Đã hủy":
                    job.remote_id = None
                    job.cloud = replace(job.cloud, idempotency_key=uuid.uuid4().hex)
                job.cloud = replace(job.cloud, approved_at=time.time())
            job.cancel.clear()
            job.status, job.error, job.progress = "Đang chờ", "", 0
            self.refresh_jobs()
            self.start_next()

    def load_job(self):
        job = self.selected_job()
        if job:
            self.open_video(job.meta.path, restored=job)

    def closeEvent(self, event):
        if self.active_job and not self.closing:
            answer = QMessageBox.question(self, "Đóng ứng dụng", "Hủy tác vụ đang chạy và đóng ứng dụng?",
                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                       QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        self.closing = True
        self.stop_play()
        self.pending_preview = None
        for job in self.jobs:
            job.cancel.set()
            if job.status == "Đang chờ":
                job.status = "Đã hủy"
        if self.tasks or self.export_pool.activeThreadCount():
            self.statusBar().showMessage("Đang hủy và dọn tệp tạm trước khi đóng…")
            event.ignore()
            QTimer.singleShot(200, self.close)
            return
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Xóa logo video")
    window = Window()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
