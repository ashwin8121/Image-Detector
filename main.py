from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QSplitter, QFrame
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QPixmap, QFont, QIcon
from matplotlib import pyplot as plt
from skimage.transform import resize
from tensorflow.keras.models import load_model
import numpy as np
import sys
from tensorflow.keras.datasets import cifar10
# from tensorflow.keras.layers import MaxPooling2D, Conv2D, Dense, Dropout, Flatten
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.activations import relu, softmax
# from tensorflow.keras.losses import SparseCategoricalCrossentropy
print(1)
img_frm_txt = """
            Select any one image among the following:
\t    -->Aeroplane
\t    -->Ship
\t    -->Bird
\t    -->Automobile
\t    -->Truck
\t    -->Dog
\t    -->Cat
\t    -->Horse
\t    -->Deer
\t    -->Frog
"""


class Main(QWidget):
    image = None

    def __init__(self):
        self.model = load_model("model_256.h5")
        super().__init__()
        # For the Window's Basic
        self.setStyleSheet(open("SpyBot.qss").read())
        self.setWindowTitle("Image Recogniser Built On Tensorflow/Keras with CIFAR-10 Dataset")
        self.resize(QSize(640, 480))

        # For the Splitter
        self.hbox = QHBoxLayout()
        self.top = QFrame()
        self.top.resize(QSize(640, 480))
        self.top.setFrameShape(QFrame.StyledPanel)

        self.bottom = QFrame()
        self.bottom.resize(QSize(640, 480))
        self.bottom.setFrameShape(QFrame.StyledPanel)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.top)
        splitter.addWidget(self.bottom)

        self.hbox.addWidget(splitter)
        self.setLayout(self.hbox)

        # The Left Frame
        vbox1 = QVBoxLayout()
        open_bt = QPushButton("Open Image")
        open_bt.clicked.connect(self.open_image)
        self.img_frame = QLabel(img_frm_txt)
        # self.img_frame.setAlignment(Qt.AlignCenter)
        vbox1.addWidget(open_bt)
        vbox1.addWidget(self.img_frame)

        hbox2 = QHBoxLayout()
        pred_bt = QPushButton("Predict")
        pred_bt.clicked.connect(self.predict)
        canc_bt = QPushButton("Cancel")
        canc_bt.clicked.connect(self.cancel)
        hbox2.addWidget(pred_bt)
        hbox2.addWidget(canc_bt)

        vbox1.addLayout(hbox2)
        self.top.setLayout(vbox1)

        vbox2 = QVBoxLayout()

        hbox3 = QHBoxLayout()
        self.prediction_label = QLabel("No Image Selected Yet")
        self.prediction_label.setFont(QFont("courier new"))
        hbox3.addWidget(self.prediction_label)
        vbox2.addLayout(hbox3)

        hbox4 = QHBoxLayout()
        pred_graph_bt = QPushButton("Show Prediction\nDetails as a Graph")
        pred_graph_bt.clicked.connect(self.prediction_details)
        model_graph_bt = QPushButton("Show Model\nDetails as a Graph")
        model_graph_bt.clicked.connect(self.model_details)
        hbox4.addWidget(pred_graph_bt)
        hbox4.addWidget(model_graph_bt)
        vbox2.addLayout(hbox4)

        self.bottom.setLayout(vbox2)

    def open_image(self):
        try:
            file, f_type = QFileDialog.getOpenFileName(caption="Open The Image", filter="JPEG (*.jpg)", directory="C:/Users/admin/Desktop/Python")
            if file:
                self.image = plt.imread(file)
                self.img_frame.setPixmap(QPixmap(file))
                self.img_frame.setAlignment(Qt.AlignCenter)
        except Exception as excep:
            print(excep)

    def predict(self):
        self.label_dict = {
            0: 'aeroplane  ',
            1: 'automobile ',
            2: 'bird       ',
            3: 'cat        ',
            4: 'deer       ',
            5: 'dog        ',
            6: 'frog       ',
            7: 'horse      ',
            8: 'ship       ',
            9: 'truck      '
        }
        txt = ""
        self.pred_dict = {}
        try:
            self.image = resize(self.image, (32, 32, 3))
            self._pred = self.model.predict(np.array([self.image]))
            self.pred = self._pred[0]
            for i in range(len(self.pred)):  # 10
                self.pred_dict.update({self.pred[i]: i})
            self.pred = np.sort(self.pred)[::-1]
            for x in range(5):
                txt += f"{self.label_dict[self.pred_dict[self.pred[x]]].capitalize()} :{round(float(self.pred[x]), 5) * 100}%\n"
            self.prediction_label.setText(txt)
        except Exception as ex:
            print(ex)

    def cancel(self):
        self.img_frame.setText(img_frm_txt)
        self.img_frame.setAlignment(Qt.AlignLeft)
        self.img_frame.setAlignment(Qt.AlignVCenter)

    def prediction_details(self):
        try:
            x_axis = [x.strip() for x in self.label_dict.values()]
            y_axis = [x * 100 for x in self._pred[0]]
            bar = plt.bar(x_axis, y_axis)
            bar[y_axis.index(max(y_axis))].set_color('r')
            bar[y_axis.index(max(y_axis)) + 1].set_color('r')
            plt.ylim((1, 100))
            plt.xticks(rotation=30)
            plt.set_cmap('Dark2')
            plt.xlabel("Categories", labelpad=4, fontdict={'family': 'consolas', 'fontsize': 15})
            plt.show()
        except Exception as e:
            print(e)

    def model_details(self):
        try:
            (x_test, y_test) = cifar10.load_data()[1]
            print(x_test, y_test)
            acc = self.model.evaluate(x_test, y_test)
            print(acc)
        except Exception as e:
            print(e)


def start_exec():
    app = QApplication(sys.argv)
    e = Main()
    e.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    start_exec()
