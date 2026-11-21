import os
import random

import _paths  # noqa: E402  必须在 PySide2 / torch 之前导入

from PySide2.QtCore import QRegExp, QSettings, Qt, QThread, QUrl, Signal
from PySide2.QtGui import QDesktopServices, QIcon, QPixmap, QRegExpValidator
from PySide2.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from sdgen import coerce_params, get_generator, latest_image

# 界面缩放系数，由 detect_ui_scale() 在启动时决定（4K + 系统缩放 100% 时放大）
UI_SCALE = 1.0


def detect_ui_scale(app):
    """根据屏幕判断是否需要额外放大界面。

    Qt 的高分屏缩放只在操作系统设置了缩放比例时才生效；如果 4K 屏是 100% 缩放，
    逻辑分辨率仍是 3840，字就会很小。这里针对这种情况补一个放大系数。
    """
    screen = app.primaryScreen()
    if screen is None:
        return 1.0
    try:
        width = screen.geometry().width()
        dpi = screen.logicalDotsPerInch()
    except Exception:
        return 1.0
    if width >= 2400 and dpi < 110:
        return 1.30
    return 1.0


def build_style(scale=1.0):
    def px(value):
        return int(round(value * scale))

    return """
QWidget {
    background: #1b1c1f;
    color: #e4e6eb;
    font-family: "Microsoft YaHei UI", "Microsoft YaHei";
    font-size: %(base)dpx;
}
QGroupBox {
    border: 1px solid #33363b;
    border-radius: %(r1)dpx;
    margin-top: %(m1)dpx;
    padding: %(p1)dpx %(p2)dpx %(p3)dpx %(p2)dpx;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: %(l1)dpx;
    padding: 0 %(p4)dpx;
    color: #7aa2f7;
    font-weight: 600;
}
QPlainTextEdit, QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background: #24262b;
    border: 1px solid #33363b;
    border-radius: %(r2)dpx;
    padding: %(p5)dpx %(p6)dpx;
    selection-background-color: #3b6fd4;
}
QPlainTextEdit:focus, QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 1px solid #5a86e8;
}
QPushButton {
    background: #2a2d33;
    border: 1px solid #3a3d44;
    border-radius: %(r2)dpx;
    padding: %(p7)dpx %(p8)dpx;
}
QPushButton:hover { background: #33373e; }
QPushButton:pressed { background: #262a30; }
QPushButton#generate {
    background: #3b6fd4;
    border: none;
    border-radius: %(r3)dpx;
    font-size: %(btn)dpx;
    font-weight: 700;
}
QPushButton#generate:hover { background: #4a7ee8; }
QPushButton#generate:disabled { background: #33363b; color: #7a7f88; }
QProgressBar {
    background: #24262b;
    border: 1px solid #33363b;
    border-radius: %(r2)dpx;
    height: %(bar)dpx;
    text-align: center;
    font-size: %(small)dpx;
}
QProgressBar::chunk { background: #3b6fd4; border-radius: %(r4)dpx; }
QLabel#title { font-size: %(title)dpx; font-weight: 700; color: #f2f4f8; }
QLabel#subtitle { color: #8b9099; font-size: %(sub)dpx; }
QLabel#image {
    background: #101114;
    border: 1px solid #33363b;
    border-radius: %(r5)dpx;
    color: #6b7078;
}
QLabel#hint { color: #8b9099; font-size: %(small)dpx; }
QLabel#status { color: #9aa0a8; }
""" % {
        "base": px(14),
        "title": px(28),
        "sub": px(13),
        "small": px(12),
        "btn": px(18),
        "bar": px(18),
        "r1": px(8),
        "r2": px(6),
        "r3": px(8),
        "r4": px(5),
        "r5": px(10),
        "m1": px(16),
        "p1": px(14),
        "p2": px(12),
        "p3": px(12),
        "p4": px(6),
        "p5": px(8),
        "p6": px(10),
        "p7": px(9),
        "p8": px(14),
        "l1": px(12),
    }


class GenerateWorker(QThread):
    """后台生成线程。模型只在第一次生成时加载，之后常驻。"""

    ok = Signal(str, int)
    fail = Signal(str)
    step = Signal(int, int)

    def __init__(self, params, seed, negative_prompt, parent=None):
        super().__init__(parent)
        self.params = params
        self.seed = seed
        self.negative_prompt = negative_prompt

    def run(self):
        try:
            generator = get_generator()
            path, seed = generator.generate(
                self.params,
                seed=self.seed,
                negative_prompt=self.negative_prompt,
                progress=lambda s, t: self.step.emit(s, t),
            )
            self.ok.emit(path, seed)
        except Exception:
            import traceback

            self.fail.emit(traceback.format_exc())


class MainWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stable Diffusion v1 — GUI")
        self.settings = QSettings("stable-diffusion-gui", "gui")
        self.setStyleSheet(build_style(UI_SCALE))
        self.setMinimumSize(int(1080 * UI_SCALE), int(740 * UI_SCALE))

        self.worker = None
        self._pixmap = None

        self._build()
        self._restore_geometry()
        self.load_image(latest_image())

    # ---------- 构建界面 ----------
    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(18)

        left = QVBoxLayout()
        left.setSpacing(8)
        self.image = QLabel("生成后在这里预览")
        self.image.setObjectName("image")
        self.image.setAlignment(Qt.AlignCenter)
        self.image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image.setMinimumSize(int(520 * UI_SCALE), int(520 * UI_SCALE))
        left.addWidget(self.image, 1)

        self.file_label = QLabel("")
        self.file_label.setObjectName("hint")
        left.addWidget(self.file_label)
        root.addLayout(left, 1)

        panel_widget = QWidget()
        panel_widget.setFixedWidth(int(440 * UI_SCALE))
        panel = QVBoxLayout(panel_widget)
        panel.setContentsMargins(0, 0, 0, 0)
        panel.setSpacing(10)

        title = QLabel("Stable Diffusion v1")
        title.setObjectName("title")
        subtitle = QLabel("本地文生图 · 模型常驻显存")
        subtitle.setObjectName("subtitle")
        panel.addWidget(title)
        panel.addWidget(subtitle)

        gb_prompt = QGroupBox("提示词")
        v1 = QVBoxLayout(gb_prompt)
        v1.setSpacing(6)
        self.prompt = QPlainTextEdit()
        self.prompt.setPlainText("a cat")
        self.prompt.setFixedHeight(int(92 * UI_SCALE))
        v1.addWidget(self.prompt)

        neg_label = QLabel("负面提示词")
        neg_label.setObjectName("hint")
        v1.addWidget(neg_label)
        self.negative = QPlainTextEdit()
        self.negative.setPlainText("blurry, lowres, bad anatomy, watermark")
        self.negative.setFixedHeight(int(64 * UI_SCALE))
        v1.addWidget(self.negative)
        panel.addWidget(gb_prompt)

        gb_param = QGroupBox("参数")
        grid = QGridLayout(gb_param)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        grid.setColumnStretch(1, 1)

        self.seed_edit = QLineEdit()
        self.seed_edit.setPlaceholderText("留空 = 随机")
        self.seed_edit.setValidator(QRegExpValidator(QRegExp("[0-9]{0,9}")))
        seed_btn = QPushButton("随机")
        seed_btn.setFixedWidth(int(60 * UI_SCALE))
        seed_btn.clicked.connect(self._fill_random_seed)

        seed_row = QHBoxLayout()
        seed_row.setSpacing(6)
        seed_row.addWidget(self.seed_edit, 1)
        seed_row.addWidget(seed_btn)

        self.batch = self._int_spin(1, 16, 1)
        self.n_iter = self._int_spin(1, 100, 1)
        self.steps = self._int_spin(1, 200, 30)
        self.scale = self._float_spin(1.0, 30.0, 9.0, 0.5)
        self.width = self._int_spin(64, 1024, 512, 64)
        self.height = self._int_spin(64, 1024, 512, 64)
        self.factor = QComboBox()
        self.factor.addItems(["8", "4", "16"])

        rows = [
            ("种子", seed_row),
            ("批次大小", self.batch),
            ("迭代次数", self.n_iter),
            ("采样步数", self.steps),
            ("CFG 引导", self.scale),
            ("宽度", self.width),
            ("高度", self.height),
            ("下采样因子", self.factor),
        ]
        for row, (label_text, widget) in enumerate(rows):
            grid.addWidget(QLabel(label_text), row, 0)
            if isinstance(widget, QHBoxLayout):
                grid.addLayout(widget, row, 1)
            else:
                grid.addWidget(widget, row, 1)
        panel.addWidget(gb_param)

        self.generate_btn = QPushButton("生 成")
        self.generate_btn.setObjectName("generate")
        self.generate_btn.setMinimumHeight(int(48 * UI_SCALE))
        self.generate_btn.clicked.connect(self.click_generate)
        panel.addWidget(self.generate_btn)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setTextVisible(True)
        panel.addWidget(self.progress)

        self.status = QLabel("就绪")
        self.status.setObjectName("status")
        panel.addWidget(self.status)

        bottom = QHBoxLayout()
        bottom.setSpacing(8)
        for text, slot in (
            ("打开目录", self.openfile),
            ("重新载入", self.update_img),
            ("退出", self.close),
        ):
            btn = QPushButton(text)
            btn.clicked.connect(slot)
            bottom.addWidget(btn)
        panel.addLayout(bottom)
        panel.addStretch(1)

        root.addWidget(panel_widget, 0)

    @staticmethod
    def _int_spin(low, high, value, step=1):
        spin = QSpinBox()
        spin.setRange(low, high)
        spin.setSingleStep(step)
        spin.setValue(value)
        return spin

    @staticmethod
    def _float_spin(low, high, value, step):
        spin = QDoubleSpinBox()
        spin.setRange(low, high)
        spin.setSingleStep(step)
        spin.setDecimals(1)
        spin.setValue(value)
        return spin

    # ---------- 窗口几何 ----------
    def _restore_geometry(self):
        saved = self.settings.value("window/geometry")
        if saved is not None and self.restoreGeometry(saved) and self._on_some_screen():
            return
        self.resize(int(1320 * UI_SCALE), int(880 * UI_SCALE))
        self._center_on_screen()

    def _on_some_screen(self):
        """窗口中心必须落在某块屏幕的可用区域内，否则视为无效几何。"""
        center = self.frameGeometry().center()
        for screen in QApplication.screens():
            if screen.availableGeometry().contains(center):
                return True
        return False

    def _center_on_screen(self):
        available = QApplication.primaryScreen().availableGeometry()
        frame = self.frameGeometry()
        frame.moveCenter(available.center())
        self.move(frame.topLeft())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale()

    # ---------- 图片 ----------
    def load_image(self, path):
        if not path or not os.path.exists(path):
            return
        self._pixmap = QPixmap(path)
        self.file_label.setText(os.path.basename(path))
        self._rescale()

    def _rescale(self):
        if self._pixmap is None or self._pixmap.isNull():
            return
        self.image.setPixmap(
            self._pixmap.scaled(self.image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def update_img(self):
        self.load_image(latest_image())

    # ---------- 参数 ----------
    def _fill_random_seed(self):
        self.seed_edit.setText(str(random.randint(1, 999999999)))

    def collect_params(self):
        return coerce_params(
            [
                self.batch.value(),
                float(self.scale.value()),
                self.steps.value(),
                self.height.value(),
                self.width.value(),
                int(self.factor.currentText()),
                self.n_iter.value(),
                self.prompt.toPlainText().strip(),
            ]
        )

    # ---------- 生成 ----------
    def click_generate(self):
        if self.worker is not None and self.worker.isRunning():
            return

        try:
            params = self.collect_params()
        except SystemExit as exc:
            QMessageBox.warning(self, "参数错误", str(exc))
            return

        seed_text = self.seed_edit.text().strip()
        seed = int(seed_text) if seed_text else None

        self.generate_btn.setEnabled(False)
        self.generate_btn.setText("生成中…")
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        if get_generator().loaded:
            self.status.setText("生成中…")
        else:
            self.status.setText("首次加载模型（8–12 秒）…")

        self.worker = GenerateWorker(params, seed, self.negative.toPlainText().strip(), self)
        self.worker.step.connect(self.on_step)
        self.worker.ok.connect(self.on_done)
        self.worker.fail.connect(self.on_fail)
        self.worker.start()

    def on_step(self, step, total):
        if self.progress.maximum() != total:
            self.progress.setRange(0, total)
        self.progress.setValue(step)
        self.status.setText("采样 %d / %d" % (step, total))

    def on_done(self, path, seed):
        self.progress.setVisible(False)
        self.progress.setRange(0, 100)
        self.generate_btn.setEnabled(True)
        self.generate_btn.setText("生 成")
        self.load_image(path)
        self.status.setText("完成 · 种子 %d" % seed)

    def on_fail(self, message):
        self.progress.setVisible(False)
        self.generate_btn.setEnabled(True)
        self.generate_btn.setText("生 成")
        self.status.setText("失败")
        print(message)
        QMessageBox.critical(self, "生成失败", message[-2000:])

    # ---------- 其它 ----------
    def openfile(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(_paths.samples_dir())))

    def closeEvent(self, event):
        self.settings.setValue("window/geometry", self.saveGeometry())
        if self.worker is not None and self.worker.isRunning():
            self.worker.wait(3000)
        super().closeEvent(event)


def main():
    os.makedirs(str(_paths.OUTPUT_DIR), exist_ok=True)

    try:
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
    except Exception:
        pass
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication([])

    global UI_SCALE
    UI_SCALE = detect_ui_scale(app)

    window = MainWindow()
    icon_path = _paths.APP_DIR / "icon.ico"
    if icon_path.exists():
        window.setWindowIcon(QIcon(str(icon_path)))
    window.show()
    app.exec_()


if __name__ == "__main__":
    main()