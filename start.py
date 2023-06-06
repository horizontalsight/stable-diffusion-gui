import pickle
import subprocess
import sys

from PySide2.QtWidgets import QApplication, QMessageBox,QFileDialog
from PySide2.QtUiTools import QUiLoader
from PySide2.QtGui import QPixmap, QDesktopServices
from PySide2.QtCore import QUrl

global array
array = [0] * 8
class Stats:

    def __init__(self):
        self.ui = QUiLoader().load('ui/ui_main_start.ui')

        path = self.load_img_init()
        self.load_image(path)

        self.ui.generate.clicked.connect(self.click_generate)
        self.ui.open.clicked.connect(self.openfile)
        self.ui.reload_img.clicked.connect(self.update_img)
        self.ui.quit.clicked.connect(self.quit_the_window)

    def load_image(self, path):
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
        with open('output/path_list.txt', 'r') as f:
            lines = f.readlines()
            if not lines:
                return None
            last_line = lines[-1].strip()
            path = last_line.split("_")[0]
            return path

    def click_generate(self):
        global array
        o = False

        with open('output/main_is_working.txt', 'r') as f:
            first_line = f.readline().strip()
            if not first_line:
                print("文件中没有内容")
            elif first_line == "main's work is done":
                #更改txt文件中的内容，把"main's work is done"改为"main is working"
                with open('output/main_is_working.txt', 'w') as f:
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

            script_path = "m.py"
            process = subprocess.Popen(['python', script_path, str(array)])
            self.process = process

            o = False

        if o == False:
            print("no generate")

    def openfile(self):
        folder_path = 'output/samples'
        QDesktopServices.openUrl(QUrl.fromLocalFile(folder_path))
        print("openfile")

    def update_img(self):
        path = self.load_img_init()
        self.load_image(path)

    def quit_the_window(self):
        self.ui.close()

def main():
    # 确保第一次的main能够执行
    with open('output/main_is_working.txt', 'w') as f:
        f.write("main's work is done")

    app = QApplication([])
    stats = Stats()
    stats.ui.show()
    app.exec_()

if __name__ == "__main__":
    main()