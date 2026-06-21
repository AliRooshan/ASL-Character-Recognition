import sys
import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFrame, QLabel, QPushButton, QSlider, QTextEdit, QSizePolicy
)
from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QLinearGradient, QTextCursor

from src.camera_worker import CameraWorker
from src.word_builder import WordBuilder

try:
    from PyQt6.QtTextToSpeech import QTextToSpeech
except ImportError:
    QTextToSpeech = None


class CircularProgressRing(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.letter = "Waiting"
        self.progress = 0.0
        self.setMinimumSize(100, 100)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_data(self, letter, progress):
        self.letter = letter
        self.progress = progress
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        size = min(self.width(), self.height())
        margin = 12
        rect = QRectF(margin, margin, size - 2 * margin, size - 2 * margin)
        rect.moveCenter(QPointF(self.width() / 2.0, self.height() / 2.0))

        track_pen = QPen(QColor("#32325d"), 6)
        painter.setPen(track_pen)
        painter.drawEllipse(rect)

        if self.progress > 0.0 and self.letter not in ("Waiting", "Error", "Waiting for hand"):
            start_angle = 90 * 16
            span_angle = int(-360 * self.progress * 16)

            gradient = QLinearGradient(rect.topLeft(), rect.bottomRight())
            gradient.setColorAt(0.0, QColor("#ec4899"))
            gradient.setColorAt(1.0, QColor("#8b5cf6"))

            progress_pen = QPen(gradient, 6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            painter.setPen(progress_pen)
            painter.drawArc(rect, start_angle, span_angle)

        if "Waiting" in self.letter or "hand" in self.letter:
            text_color = QColor("#94a3b8")
            font_size = 11
            display_char = "..."
        elif self.letter == "space":
            text_color = QColor("#ec4899")
            font_size = 14
            display_char = "SPACE"
        elif self.letter == "del":
            text_color = QColor("#ef4444")
            font_size = 14
            display_char = "DEL"
        else:
            text_color = QColor("#ec4899")
            font_size = 32
            display_char = self.letter

        painter.setPen(text_color)
        font = QFont("Segoe UI", font_size, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, display_char)


class ASLRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ASL Sign Language Recognition System")
        self.resize(1080, 750)
        self.setMinimumSize(960, 680)

        self.builder = WordBuilder()
        self.worker = None
        self.stream_paused = False

        self.tts = None
        if QTextToSpeech:
            try:
                self.tts = QTextToSpeech(self)
            except Exception as e:
                print(f"Warning: Native TTS module initialization failed: {e}")

        self.init_ui()
        self.apply_stylesheet()
        self.toggle_stream()

    def init_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        left_panel = QVBoxLayout()
        left_panel.setSpacing(15)

        header_layout = QVBoxLayout()
        header_layout.setSpacing(4)
        title_label = QLabel("ASL Sign Translator", self)
        title_label.setObjectName("titleLabel")
        subtitle_label = QLabel("Real-time Sign Language Recognition using Computer Vision", self)
        subtitle_label.setObjectName("subtitleLabel")
        header_layout.addWidget(title_label)
        header_layout.addWidget(subtitle_label)
        left_panel.addLayout(header_layout)

        self.webcam_card = QFrame(self)
        self.webcam_card.setObjectName("webcamCard")
        webcam_layout = QVBoxLayout(self.webcam_card)
        webcam_layout.setContentsMargins(8, 8, 8, 8)

        self.webcam_viewport = QLabel(self)
        self.webcam_viewport.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.webcam_viewport.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.show_placeholder_frame("Webcam ready. Press 'Start Stream' to capture.")
        webcam_layout.addWidget(self.webcam_viewport)
        left_panel.addWidget(self.webcam_card, 4)

        status_layout = QHBoxLayout()
        status_layout.addWidget(QLabel("Webcam Status:", self))
        self.status_badge = QLabel("WAITING FOR HAND", self)
        self.status_badge.setObjectName("statusBadge")
        self.status_badge.setProperty("active", "false")
        self.status_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_layout.addWidget(self.status_badge)
        status_layout.addStretch()

        self.btn_skeleton = QPushButton("Hide Skeleton", self)
        self.btn_skeleton.setCheckable(True)
        self.btn_skeleton.setChecked(True)
        self.btn_skeleton.setObjectName("btnSkeleton")
        self.btn_skeleton.clicked.connect(self.toggle_skeleton)
        status_layout.addWidget(self.btn_skeleton)

        self.btn_toggle_stream = QPushButton("Pause Stream", self)
        self.btn_toggle_stream.setObjectName("btnToggleStream")
        self.btn_toggle_stream.clicked.connect(self.toggle_stream)
        status_layout.addWidget(self.btn_toggle_stream)

        left_panel.addLayout(status_layout)

        main_layout.addLayout(left_panel, 3)

        right_panel = QVBoxLayout()
        right_panel.setSpacing(16)

        signs_card = QFrame(self)
        signs_card.setObjectName("signsCard")
        signs_layout = QVBoxLayout(signs_card)
        signs_layout.setContentsMargins(8, 8, 8, 8)

        self.signs_image = QLabel(self)
        self.signs_image.setAlignment(Qt.AlignmentFlag.AlignCenter)

        signs_path = "C:/Users/user/OneDrive/Desktop/CV Project/assets/signs.png"
        if not os.path.exists(signs_path):
            signs_path = os.path.join(os.getcwd(), "signs.png")

        if os.path.exists(signs_path):
            pixmap = QPixmap(signs_path)
            self.signs_image.setPixmap(pixmap.scaled(
                480, 360,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))
        else:
            self.signs_image.setText("signs.png not found")
            self.signs_image.setStyleSheet("color: #71717a;")

        signs_layout.addWidget(self.signs_image)
        right_panel.addWidget(signs_card, 4)

        unified_card = QFrame(self)
        unified_card.setObjectName("unifiedCard")
        unified_layout = QVBoxLayout(unified_card)
        unified_layout.setContentsMargins(16, 16, 16, 16)
        unified_layout.setSpacing(12)

        pred_layout = QHBoxLayout()
        pred_layout.setSpacing(16)

        settings_layout = QVBoxLayout()
        settings_layout.setSpacing(10)

        self.lbl_predictions = QLabel("Character Translations", self)
        self.lbl_predictions.setObjectName("predictionsHeading")
        settings_layout.addWidget(self.lbl_predictions)

        cooldown_layout = QHBoxLayout()
        cooldown_layout.setSpacing(10)

        self.lbl_cooldown = QLabel("Cooldown: 30", self)
        self.lbl_cooldown.setObjectName("cooldownLabel")
        cooldown_layout.addWidget(self.lbl_cooldown)

        self.slider_cooldown = QSlider(Qt.Orientation.Horizontal, self)
        self.slider_cooldown.setRange(15, 60)
        self.slider_cooldown.setValue(30)
        self.slider_cooldown.valueChanged.connect(self.on_cooldown_slider_changed)
        cooldown_layout.addWidget(self.slider_cooldown)

        settings_layout.addLayout(cooldown_layout)

        self.progress_ring = CircularProgressRing(self)
        self.progress_ring.setFixedSize(100, 100)
        pred_layout.addLayout(settings_layout, 1)
        pred_layout.addWidget(self.progress_ring, 0, Qt.AlignmentFlag.AlignVCenter)
        unified_layout.addLayout(pred_layout)

        self.text_edit = QTextEdit(self)
        self.text_edit.setObjectName("sentenceText")
        self.text_edit.textChanged.connect(self.sync_builder_text)
        unified_layout.addWidget(self.text_edit)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)

        self.btn_speak = QPushButton("Speak", self)
        self.btn_speak.setObjectName("btnSpeak")
        self.btn_speak.clicked.connect(self.action_speak)
        if not self.tts:
            self.btn_speak.setEnabled(False)
            self.btn_speak.setToolTip("TTS engine not available on this platform.")

        self.btn_clear = QPushButton("Clear", self)
        self.btn_clear.setObjectName("btnClear")
        self.btn_clear.clicked.connect(self.action_clear)

        btn_layout.addWidget(self.btn_speak)
        btn_layout.addWidget(self.btn_clear)
        unified_layout.addLayout(btn_layout)

        right_panel.addWidget(unified_card, 1)
        main_layout.addLayout(right_panel, 2)

    def apply_stylesheet(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #121225;
            }
            QLabel {
                color: #e2e8f0;
                font-family: 'Segoe UI', sans-serif;
                font-size: 13px;
            }
            QFrame#webcamCard, QFrame#unifiedCard, QFrame#signsCard {
                background-color: #1b1b3a;
                border: 1px solid #32325d;
                border-radius: 12px;
            }
            QLabel#titleLabel {
                color: #ffffff;
                font-size: 24px;
                font-weight: bold;
            }
            QLabel#subtitleLabel {
                color: #94a3b8;
                font-size: 12px;
            }
            QLabel#cardTitle {
                color: #ec4899;
                font-size: 15px;
                font-weight: bold;
            }
            QLabel#predictionsHeading {
                color: #ffffff;
                font-size: 24px;
                font-weight: bold;
            }
            QLabel#cooldownLabel {
                font-weight: bold;
                color: #e2e8f0;
            }
            QLabel#statusBadge {
                font-size: 11px;
                font-weight: bold;
                border-radius: 6px;
                padding: 4px 10px;
                font-family: 'Segoe UI', sans-serif;
            }
            QLabel#statusBadge[active="true"] {
                background-color: rgba(16, 185, 129, 0.15);
                color: #34d399;
                border: 1px solid #10b981;
            }
            QLabel#statusBadge[active="false"] {
                background-color: rgba(239, 68, 68, 0.15);
                color: #f87171;
                border: 1px solid #ef4444;
            }
            QTextEdit#sentenceText {
                background-color: #0f0f1e;
                color: #ffffff;
                border: 1px solid #32325d;
                border-radius: 8px;
                font-size: 18px;
                font-family: 'Consolas', 'Courier New', monospace;
                padding: 10px;
            }
            QTextEdit#sentenceText:focus {
                border: 1px solid #ec4899;
            }
            QPushButton {
                color: #ffffff;
                background-color: #312e81;
                border: 1px solid #4338ca;
                font-weight: bold;
                border-radius: 8px;
                padding: 9px 18px;
                font-family: 'Segoe UI', sans-serif;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #3730a3;
            }
            QPushButton:pressed {
                background-color: #1e1b4b;
            }
            QPushButton#btnSpeak {
                background-color: #10b981;
                border: none;
            }
            QPushButton#btnSpeak:hover {
                background-color: #059669;
            }
            QPushButton#btnSpeak:pressed {
                background-color: #047857;
            }
            QPushButton#btnClear {
                background-color: #ef4444;
                border: none;
            }
            QPushButton#btnClear:hover {
                background-color: #dc2626;
            }
            QPushButton#btnClear:pressed {
                background-color: #b91c1c;
            }
            QSlider::groove:horizontal {
                border: 1px solid #32325d;
                height: 6px;
                background: #25254b;
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #ec4899;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #ffffff;
                border: 1px solid #ec4899;
                width: 14px;
                height: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
            QSlider::handle:horizontal:hover {
                background: #fdf2f8;
            }
            QPushButton#btnSkeleton:checked {
                background-color: #ec4899;
                border: none;
            }
            QPushButton#btnSkeleton:checked:hover {
                background-color: #db2777;
            }
            QPushButton#btnSkeleton:checked:pressed {
                background-color: #be185d;
            }
        """)

    def show_placeholder_frame(self, text):
        pixmap = QPixmap(640, 480)
        pixmap.fill(QColor("#0f0f1e"))
        painter = QPainter(pixmap)
        painter.setPen(QColor("#94a3b8"))
        painter.setFont(QFont("Segoe UI", 12))
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, text)
        painter.end()
        self.webcam_viewport.setPixmap(pixmap)

    @pyqtSlot(QImage, str, bool)
    def update_frame(self, q_img, prediction, hand_detected):
        if self.stream_paused:
            return

        pixmap = QPixmap.fromImage(q_img)
        self.webcam_viewport.setPixmap(pixmap.scaled(
            self.webcam_viewport.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))

        if hand_detected:
            self.status_badge.setText("HAND DETECTED")
            self.status_badge.setProperty("active", "true")
            self.webcam_card.setStyleSheet("QFrame#webcamCard { border: 1px solid #10b981; }")
        else:
            self.status_badge.setText("WAITING FOR HAND")
            self.status_badge.setProperty("active", "false")
            self.webcam_card.setStyleSheet("QFrame#webcamCard { border: 1px solid #32325d; }")
            
        self.status_badge.style().unpolish(self.status_badge)
        self.status_badge.style().polish(self.status_badge)

        progress = 0.0
        if hand_detected:
            self.builder.text = self.text_edit.toPlainText()
            new_text = self.builder.update(prediction)
            
            if self.text_edit.toPlainText() != new_text:
                self.text_edit.setPlainText(new_text)
                cursor = self.text_edit.textCursor()
                cursor.movePosition(QTextCursor.MoveOperation.End)
                self.text_edit.setTextCursor(cursor)

            if len(self.builder.buffer) >= self.builder.buffer.maxlen:
                progress = min(1.0, self.builder.frame_count / self.builder.cooldown)

        self.progress_ring.set_data(prediction, progress)

    @pyqtSlot(str)
    def handle_error(self, message):
        self.show_placeholder_frame(f"Error: {message}")
        self.status_badge.setText("ERROR")
        self.status_badge.setProperty("active", "false")
        self.status_badge.style().unpolish(self.status_badge)
        self.status_badge.style().polish(self.status_badge)
        self.btn_toggle_stream.setText("Resume Stream")
        self.stream_paused = True

    def toggle_stream(self):
        if self.worker and self.worker.isRunning():
            self.btn_toggle_stream.setText("Resume Stream")
            self.stream_paused = True
            self.worker.stop()
            self.worker.wait()
            self.worker = None
            self.show_placeholder_frame("Camera paused. Press 'Resume Stream' to start.")
            self.status_badge.setText("PAUSED")
            self.status_badge.setProperty("active", "false")
            self.status_badge.style().unpolish(self.status_badge)
            self.status_badge.style().polish(self.status_badge)
        else:
            self.btn_toggle_stream.setText("Pause Stream")
            self.stream_paused = False
            self.worker = CameraWorker()
            
            self.worker.frame_ready.connect(self.update_frame)
            self.worker.error_occurred.connect(self.handle_error)

            self.worker.set_model("svm")
            self.worker.set_min_detection_confidence(0.70)
            self.worker.set_draw_landmarks(self.btn_skeleton.isChecked())
            self.worker.start()

    def sync_builder_text(self):
        self.builder.text = self.text_edit.toPlainText()

    def on_cooldown_slider_changed(self, value):
        self.builder.set_cooldown(value)
        self.lbl_cooldown.setText(f"Cooldown: {value}")

    def toggle_skeleton(self):
        enabled = self.btn_skeleton.isChecked()
        if enabled:
            self.btn_skeleton.setText("Hide Skeleton")
        else:
            self.btn_skeleton.setText("Show Skeleton")
        if self.worker:
            self.worker.set_draw_landmarks(enabled)

    def action_clear(self):
        self.text_edit.setPlainText("")
        self.builder.text = ""
        self.builder.buffer.clear()
        self.builder.last_char = None
        self.builder.frame_count = 0
        self.progress_ring.set_data("Waiting...", 0.0)

    def action_speak(self):
        text = self.text_edit.toPlainText().strip()
        if text and self.tts:
            self.tts.say(text)

    def closeEvent(self, event):
        if self.worker:
            self.worker.stop()
            if not self.worker.wait(1500):
                self.worker.terminate()
                self.worker.wait()
            self.worker = None
        event.accept()


def run():
    app = QApplication(sys.argv)
    window = ASLRecognitionApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run()