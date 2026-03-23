import os
import sys
import cv2
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, \
    QMessageBox
from PyQt5.QtGui import QFont, QPixmap, QIcon
from PyQt5.QtCore import Qt
import tensorflow as tf

OUTPUT_DIR = 'tu'
NON_RECYCLABLE_DIR = 'non_recyclable'
RECYCLABLE_DIR = 'recyclable'
HAZARDOUS_DIR = 'hazardous'
OTHER_DIR = 'other'

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon("./9.jpg"))
        self.setWindowTitle('垃圾分类')

        self.model = tf.keras.models.load_model("./mobilenetv2_laji.h5")
        self.to_predict_name = ("6.png")
        self.to_predict_images = []
        self.class_names = ['一次性杯子', '卫生纸', '口罩', '指甲油','易拉罐','杀虫剂','果皮','水果','瓶子','纸袋','过期药物','食物']
        self.resize(700, 500)
        self.initUI()

    def initUI(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout()

        font = QFont('楷体', 25)

        left_widget = QWidget()
        left_layout = QVBoxLayout()
        img_title = QLabel("图像")
        img_title.setFont(font)
        img_title.setAlignment(Qt.AlignCenter)
        self.img_label = QLabel()
        self.update_image()
        left_layout.addWidget(img_title)
        left_layout.addWidget(self.img_label, 1, Qt.AlignCenter)
        left_widget.setLayout(left_layout)

        right_widget = QWidget()
        right_layout = QVBoxLayout()
        btn_change = QPushButton("选择图像")
        btn_change.clicked.connect(self.change_images)
        btn_change.setFont(font)
        btn_change.setStyleSheet("background-color: white;")
        btn_predict = QPushButton("识别")
        btn_predict.setFont(font)
        btn_predict.clicked.connect(self.predict_images)
        btn_predict.setStyleSheet("background-color: white;")
        self.btn_classify = QPushButton("垃圾分类")
        self.btn_classify.setFont(font)
        self.btn_classify.setEnabled(True)
        self.btn_classify.clicked.connect(self.classify_images)
        self.btn_classify.setStyleSheet("background-color: white;")
        self.btn_save = QPushButton("保存")
        self.btn_save.setFont(font)
        self.btn_save.setEnabled(True)
        self.btn_save.clicked.connect(self.save_image)
        self.btn_save.setStyleSheet("background-color: white;")

        label_result = QLabel('垃圾分类结果')
        label_result.setFont(QFont('楷体', 28))
        self.result = QLabel('待识别')
        self.result.setFont(QFont('楷体', 24))
        right_layout.addStretch()
        right_layout.addWidget(label_result, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addWidget(self.result, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addWidget(btn_change)
        right_layout.addWidget(btn_predict)
        right_layout.addWidget(self.btn_classify)
        right_layout.addWidget(self.btn_save)
        right_layout.addStretch()
        right_widget.setLayout(right_layout)

        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget)
        self.setLayout(main_layout)



    def change_images(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        if file_dialog.exec_():
            files = file_dialog.selectedFiles()
            self.to_predict_images = files
            self.update_image()
            self.btn_save.setEnabled(True)

    def update_image(self):
        if not self.to_predict_images:
            self.cover_image_path = "6.png"
            self.img_label.setPixmap(QPixmap(self.cover_image_path))
        else:
            img_path = self.to_predict_images[0]
            img = cv2.imread(img_path)
            img = cv2.resize(img, (224, 224))
            cv2.imwrite("laji1/5.jpg", img)
            self.img_label.setPixmap(QPixmap("laji1/5.jpg"))

    def create_directory(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def predict_image(self, img):
        img = cv2.resize(img, (224, 224))
        outputs = self.model.predict(np.expand_dims(img, axis=0))
        print(outputs)
        result_index = np.argmax(outputs)
        print(result_index)
        result = self.class_names[result_index]
        return result

    def predict_images(self):
        results = []
        for img_path in self.to_predict_images:
            img = cv2.imread(img_path)
            result = self.predict_image(img)
            results.append((os.path.basename(img_path), result))
            print(img_path, " is ", result)
            self.result.setText(result)
            if result in ['果皮', '水果', '食物']:
                self.btn_classify.setEnabled(True)
            elif result in ['易拉罐', '纸袋', '瓶子']:
                self.btn_classify.setEnabled(True)
            else:
                self.btn_classify.setEnabled(True)

        QMessageBox.about(self, '识别结果', '图像识别结果是：{}'.format(results))

    def classify_images(self):
        results = []
        for img_path in self.to_predict_images:
            img = cv2.imread(img_path)
            result = self.predict_image(img)
            if result in ['果皮', '水果', '食物']:
                results.append((result, result + '属于不可回收垃圾'))
                self.create_directory(NON_RECYCLABLE_DIR)
                cv2.imwrite(os.path.join(NON_RECYCLABLE_DIR, os.path.basename(img_path)), img)
            elif result in ['瓶子', '纸袋', '易拉罐']:
                results.append((result, result + '属于可回收垃圾'))
                self.create_directory(RECYCLABLE_DIR)
                cv2.imwrite(os.path.join(RECYCLABLE_DIR, os.path.basename(img_path)), img)
            elif result in ['杀虫剂', '过期药物', '指甲油']:
                results.append((result, result + '属于有害垃圾'))
                self.create_directory(HAZARDOUS_DIR)
                cv2.imwrite(os.path.join(HAZARDOUS_DIR, os.path.basename(img_path)), img)
            else:
                results.append((result, result + '属于其他垃圾'))
                self.create_directory(OTHER_DIR)
                cv2.imwrite(os.path.join(OTHER_DIR, os.path.basename(img_path)), img)
            print(img_path, " is ", results[-1][1])
            self.result.setText(results[-1][1])
        QMessageBox.about(self, '垃圾分类结果', '图像垃圾分类结果是：{}'.format(results))

    def save_image(self):
        if self.to_predict_images:
            for img_path in self.to_predict_images:
                img = cv2.imread(img_path)
                result = self.predict_image(img)
                if result in ['果皮', '水果', '食物']:
                    directory = os.path.join(OUTPUT_DIR, NON_RECYCLABLE_DIR)
                elif result in ['瓶子', '纸袋', '易拉罐']:
                    directory = os.path.join(OUTPUT_DIR, RECYCLABLE_DIR)
                elif result in ['杀虫剂', '过期药物', '指甲油']:
                    directory = os.path.join(OUTPUT_DIR, HAZARDOUS_DIR)
                else:
                    directory = os.path.join(OUTPUT_DIR, OTHER_DIR)
                self.create_directory(directory)
                cv2.imwrite(os.path.join(directory, os.path.basename(img_path)), img)
            QMessageBox.about(self, '保存结果', '图像已保存到相应的文件夹')
        else:
            QMessageBox.about(self, '保存结果', '请选择图像以保存')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    x = MainWindow()
    x.show()
    sys.exit(app.exec_())