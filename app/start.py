import os
import subprocess
import sys

import _paths  # noqa: E402  必须在 PySide2 / transformers 之前导入

from PySide2.QtWidgets import QApplication
from PySide2.QtUiTools import QUiLoader
from PySide2.QtGui import QPixmap, QDesktopServices
from PySide2.QtCore import QUrl


global array
array = [0] * 8


class Stats:

    def __init__(self):
        self.ui = QUiLoader().load(str(_paths.UI_FILE))

        path = self.load_img_init()
        self.load_image(path)

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
            label.setPixmap(pixmap.scaled(pixmap.width() * ratio, pixmap.height() * ratio))

    def read_text(self, control_name):
        control = getattr(self.ui, control_name)
        text = control.toPlainText()
        return text

    def load_img_init(self):
        if not _paths.PATH_LIST.exists():
            return None
        with open(str(_paths.PATH_LIST), "r") as f:
            lines = f.readlines()
        if not lines:
            return None
        last_line = lines[-1].strip()
        path = last_line.split("_")[0]
        return path or None

    def click_generate(self):
        global array
        o = False

        with open(str(_paths.STATE_FILE), "r") as f:
            first_line = f.readline().strip()
            if not first_line:
                print("文件中没有内容")
            elif first_line == "main's work is done":
                # 把 "main's work is done" 改成 "main is working"
                with open(str(_paths.STATE_FILE), "w") as f:
                    f.write("main is working\n")
                o = True

        if o == True:
            texts = ['T0', 'T1', 'T2', 'T4', 'T5', 'T6', 'T9', 'T10']
            array.clear()

            for i, t in enumerate(texts):
                text = self.read_text(t)

                if i == 0:  # 第一个元素
                    if '.' in text:  # 如果包含小数点，转换为浮点数
                        value = float(text)
                    else:  # 否则转换为整数
                        value = int(text)
                elif i < 7:  # t1 - t7 元素
                    value = int(text)
                else:  # 最后一个元素 t10
                    value = text

                array.append(value)

            print(array)
            print("click_generate")

            script_path = _paths.APP_DIR / "m.py"
            # 用当前解释器，避免依赖 PATH 里的 python
            self.process = subprocess.Popen([sys.executable, str(script_path), str(array)])

            o = False

        if o == False:
            print("no generate")

    def openfile(self):
        folder_path = str(_paths.SAMPLES_DIR)
        QDesktopServices.openUrl(QUrl.fromLocalFile(folder_path))
        print("openfile")

    def update_img(self):
        path = self.load_img_init()
        self.load_image(path)

    def quit_the_window(self):
        self.ui.close()


def main():
    os.makedirs(str(_paths.OUTPUT_DIR), exist_ok=True)
    # 确保第一次的 main 能够执行
    with open(str(_paths.STATE_FILE), "w") as f:
        f.write("main's work is done")

    app = QApplication([])
    stats = Stats()
    stats.ui.show()
    app.exec_()


if __name__ == "__main__":
    main()