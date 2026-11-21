import os

import _paths  # noqa: E402  必须在 PySide2 / torch 之前导入

from PySide2.QtWidgets import QApplication, QMessageBox
from PySide2.QtUiTools import QUiLoader
from PySide2.QtGui import QPixmap, QDesktopServices
from PySide2.QtCore import QUrl, QThread, Signal

from sdgen import coerce_params, latest_image, get_generator


class GenerateWorker(QThread):
    """在后台线程里生成，避免界面卡死。

    模型只在第一次生成时加载（见 sdgen.get_generator），之后复用同一份。
    """

    ok = Signal(str, int)
    fail = Signal(str)

    def __init__(self, params, parent=None):
        super().__init__(parent)
        self.params = params

    def run(self):
        try:
            generator = get_generator()
            path, seed = generator.generate(self.params)
            self.ok.emit(path, seed)
        except Exception:
            import traceback

            self.fail.emit(traceback.format_exc())


class Stats:

    def __init__(self):
        self.ui = QUiLoader().load(str(_paths.UI_FILE))
        self.worker = None

        self.load_image(latest_image())

        self.ui.generate.clicked.connect(self.click_generate)
        self.ui.open.clicked.connect(self.openfile)
        self.ui.reload_img.clicked.connect(self.update_img)
        self.ui.quit.clicked.connect(self.quit_the_window)

    def load_image(self, path):
        if not path or not os.path.exists(path):
            return
        pixmap = QPixmap(path)
        label = self.ui.label_image
        label.setPixmap(pixmap)
        label.setScaledContents(True)
        if pixmap.width() > label.width() or pixmap.height() > label.height():
            ratio = min(label.width() / pixmap.width(), label.height() / pixmap.height())
            label.setPixmap(
                pixmap.scaled(int(pixmap.width() * ratio), int(pixmap.height() * ratio))
            )

    def read_text(self, control_name):
        return getattr(self.ui, control_name).toPlainText()

    def collect_params(self):
        """把界面上的 8 个控件读成参数列表并校验。"""
        names = ['T0', 'T1', 'T2', 'T4', 'T5', 'T6', 'T9', 'T10']
        values = []
        for i, name in enumerate(names):
            text = self.read_text(name).strip()
            try:
                if i == 0:
                    value = float(text) if '.' in text else int(text)
                elif i < 7:
                    value = int(text)
                else:
                    value = text
            except ValueError:
                raise ValueError("%s 的值 %r 不是合法数字" % (name, text))
            values.append(value)
        return coerce_params(values)

    def click_generate(self):
        if self.worker is not None and self.worker.isRunning():
            print("still generating")
            return

        try:
            params = self.collect_params()
        except (ValueError, SystemExit) as exc:
            QMessageBox.warning(None, "参数错误", str(exc))
            return

        print("params:", params)

        self.ui.generate.setEnabled(False)
        self.ui.generate.setText("生成中…")

        self.worker = GenerateWorker(params, self.ui)
        self.worker.ok.connect(self.on_done)
        self.worker.fail.connect(self.on_fail)
        self.worker.start()

    def on_done(self, path, seed):
        self.load_image(path)
        self.ui.generate.setEnabled(True)
        self.ui.generate.setText("GENERATE")
        print("done: %s (seed %d)" % (path, seed))

    def on_fail(self, message):
        self.ui.generate.setEnabled(True)
        self.ui.generate.setText("GENERATE")
        print(message)
        QMessageBox.critical(None, "生成失败", message[-2000:])

    def openfile(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(_paths.SAMPLES_DIR)))
        print("openfile")

    def update_img(self):
        self.load_image(latest_image())

    def quit_the_window(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.wait(2000)
        self.ui.close()


def main():
    os.makedirs(str(_paths.OUTPUT_DIR), exist_ok=True)

    app = QApplication([])
    stats = Stats()
    stats.ui.show()
    app.exec_()


if __name__ == "__main__":
    main()