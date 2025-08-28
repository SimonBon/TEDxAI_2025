import sys

sys.path.append(".")

from pathlib import Path
from time import time

import numpy as np
from copy import deepcopy
from time import time
import torch
#import tifffile
import qimage2ndarray
from scipy.ndimage import zoom
from scipy.spatial.distance import cdist
#from einops import rearrange
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import Qt, QRect, pyqtSignal, QPoint
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QFont, QImage
from PyQt5.QtWidgets import QApplication, QMainWindow, QSpacerItem, QSizePolicy, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QTableWidget, QTableWidgetItem
from PyQt5.uic import loadUi  # Import to load .ui files
from PyQt5.QtWidgets import QApplication
import sys

from DiagnosticFISH_package.DiagnosticFISH.src.utils import get_model_dataloader, run_model
from DiagnosticFISH_package.DiagnosticFISH.src.dataset import SingleChannelDataset
from mmengine.dataset import DefaultSampler, default_collate
from torch.utils.data import DataLoader
from mmselfsup.datasets.transforms import PackSelfSupInputs
from DiagnosticFISH_package.DiagnosticFISH.src.transforms import CentralCutter, C_TensorCombiner

from PyQt5.QtGui import QFont

from pathlib import Path
BASEDIR = Path(__file__).parent.parent

# Define a font (Font Name, Font Size)
bold_font = QFont("Arial", 30)
bold_font.setBold(True)  # Make text bold

from src import randomly_place_cells, load_model

from screeninfo import get_monitors

WIDTH, HEIGHT = get_monitors()[0].width, get_monitors()[0].height
if (WIDTH / HEIGHT) > 16/9:
    print('resorting to 1920')
    WIDTH = 1920
    HEIGHT = 1056

H5_PATH = BASEDIR.joinpath('data/dataset.h5')

def resize_with_scipy(image, target_height, target_width):
    """Resize an image using scipy to a target height and width."""
    scale = (target_height / image.shape[0], target_width / image.shape[1])
    scale = scale if image.ndim == 2 else scale + (1,)
    return zoom(image, scale, order=0)

class ClickableLabel(QLabel):
    
    imageClicked = pyqtSignal(QPoint)  # Signal to emit the local position within the pixmap

    def __init__(self, parent=None):
        super(ClickableLabel, self).__init__(parent)
        self.setMouseTracking(True)  # Enable mouse tracking to receive mouse move events
        self.setAlignment(Qt.AlignCenter)  # Align the pixmap to the center of the label

    def mousePressEvent(self, event):
        """Handle the mouse press event and emit a signal if the click is within the pixmap."""
        local_click_pos = event.pos()
        print(local_click_pos)
        if self.pixmap() and self.pixmap().rect().contains(self._mapToLocalPixmap(local_click_pos)):
            self.imageClicked.emit(self._mapToLocalPixmap(local_click_pos))

    def _mapToLocalPixmap(self, position):
        """Map the position from QLabel coordinates to local pixmap coordinates."""
        pixmap_top_left = self._calculatePixmapTopLeft()
        return QPoint(position.x() - pixmap_top_left.x(), position.y() - pixmap_top_left.y())

    def _calculatePixmapTopLeft(self):
        """Calculate the top-left corner of the pixmap within the QLabel, accounting for alignment."""
        if not self.pixmap():
            return QPoint(0, 0)
        pm_width, pm_height = self.pixmap().size().width(), self.pixmap().size().height()
        lb_width, lb_height = self.size().width(), self.size().height()
        return QPoint((lb_width - pm_width) // 2, (lb_height - pm_height) // 2)

import h5py

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.setup_layout()
        self.connect_signals()
        
        self.load_h5(H5_PATH)
        self.model, self.classifier = self.get_models()
        self.started = False
        
        self.setStyleSheet("""
            QWidget {
                background-color: black;
                color: white;
            }
            QLabel {
                color: white;
            }
            QLineEdit {
                background-color: #333;
                color: white;
                border: 1px solid white;
            }
            QPushButton {
                background-color: #555;
                color: white;
                border: 1px solid white;
                padding: 5px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #777;
            }
        """)
    
    
    def get_models(self):
        
        self.pipeline = [
            C_TensorCombiner(),
            CentralCutter(size=128), 
            PackSelfSupInputs(meta_keys=[])
        ]
        
        model = get_model_dataloader(
            BASEDIR.joinpath('data/config'),
            device='cpu',
            only_model=True
        )
        
        classifier = load_model(BASEDIR.joinpath('data/classifier.pth'))        

        model.eval()
        classifier.eval()
        
        return model, classifier
    
    def generate_image(self, out_size, images, masks, targets, n_images, max_rejections):
    
        image, mask_image, idxs, positions, target = randomly_place_cells(out_size, images, masks, targets, n_images, max_rejections=max_rejections)
        
        self.dataset = SingleChannelDataset(
                H5_PATH,
                shuffle=False,
                pipeline=self.pipeline,
                channel_idx=1,
                masked_idxs=idxs)

        self.dataloader = DataLoader(
                dataset=self.dataset, 
                sampler=DefaultSampler(self.dataset, shuffle=False), 
                batch_size=16, 
                collate_fn=default_collate,
                num_workers=0
            )
        
        self.run_model()
                
        return image, mask_image, idxs, positions, target
    
        
    def load_h5(self, path):
        
        with h5py.File(path, 'r') as f:
            
            self.cells = f['FISH'][()]
            self.masks = f['NUCLEUS'][()]
            self.targets = f['N_SIGNALS'][:, 1] > 2
            

    def setup_layout(self):
        self.setWindowTitle("MainWindow")
        self.centralWidget = QWidget(self)
        self.setCentralWidget(self.centralWidget)

        self.mainHorizontalLayout = QHBoxLayout(self.centralWidget)
        self.mainHorizontalLayout.setAlignment(Qt.AlignCenter)

        self.leftContainer = QWidget(self.centralWidget)
        self.leftContainer.setMinimumSize(int(2*WIDTH//3)-10, HEIGHT)

        self.rightContainer = QWidget(self.centralWidget)
        self.rightContainer.setMinimumSize(int(WIDTH//3)-10, HEIGHT)

        self.mainHorizontalLayout.addWidget(self.leftContainer, alignment=Qt.AlignHCenter)
        self.mainHorizontalLayout.addWidget(self.rightContainer, alignment=Qt.AlignHCenter)

        # Left Layout
        leftVerticalLayout = QVBoxLayout(self.leftContainer)
        self.Image = ClickableLabel(self.leftContainer)
        self.Image.setMinimumSize(int(0.8*(2*WIDTH//3)), int(0.8*(2*WIDTH//3)))
        self.Image.setAlignment(Qt.AlignCenter)
        leftVerticalLayout.addWidget(self.Image)

        # Right Layout
        rightVerticalLayout = QVBoxLayout(self.rightContainer)

        # ⬇️ HBox for Image Size and Load Image Button
        imageInputLayout = QHBoxLayout()
        self.image_size = QLineEdit(self)
        self.image_size.setMinimumSize(200, 50)
        self.image_size.setMaximumSize(250, 50)
        self.image_size.setPlaceholderText("Bildgröße eingeben...")  # Hint text
        imageInputLayout.addWidget(self.image_size)

        self.new_image_button = self.create_button("Mikroskopiebild laden", 200, 250, 50, Qt.AlignHCenter)
        imageInputLayout.addWidget(self.new_image_button)

        rightVerticalLayout.addItem(QSpacerItem(0, 150))
        rightVerticalLayout.addLayout(imageInputLayout)

        # ⬇️ HBox for Start and Stop Buttons
        startStopLayout = QHBoxLayout()
        self.start = self.create_button("Suche beginnen...", 300, 400, 120, Qt.AlignHCenter)
        self.start.setMinimumSize(120, 80)  # Set minimum width and height
        self.start.setFont(bold_font)
        self.start.setStyleSheet("background-color: green; color: white; border-radius: 10px; font-weight: bold;")
        self.stop = self.create_button("Suche beenden...", 300, 400, 120, Qt.AlignHCenter)
        self.stop.setMinimumSize(120, 80)  # Set minimum width and height
        self.stop.setFont(bold_font)
        self.stop.setStyleSheet("background-color: red; color: white; border-radius: 10px; font-weight: bold;")
        startStopLayout.addWidget(self.start)
        startStopLayout.addWidget(self.stop)
        
        showResults = QHBoxLayout()
        self.show_results = self.create_button("Vergleiche mit KI", 200, 250, 50, Qt.AlignHCenter)
        self.show_results.setMinimumSize(120, 80)  # Set minimum width and height
        showResults.addWidget(self.show_results)

        rightVerticalLayout.addLayout(startStopLayout)
        rightVerticalLayout.addLayout(showResults)
        
        # 🔹 HBox for Summary Row (Initial Values)
        self.SummaryLayout = QVBoxLayout()
        self.SummaryLayout.setSpacing(10)  # Reduce spacing between labels (adjust as needed)
        self.SummaryLayout.setContentsMargins(0, 0, 0, 0)  # Minimize margins

        self.user_time_label = QLabel("Deine Zeit: 0s", self.rightContainer)
        self.user_time_label.setFont(bold_font)
        self.user_annot_label = QLabel("Erkannte Tumorzellen: 0\nKorrekte Anzahl: 0", self.rightContainer)
        self.user_annot_label.setFont(bold_font)
        
        self.model_time_label = QLabel("Zeit der KI: 0s", self.rightContainer)
        self.model_time_label.setFont(bold_font)
        self.model_annot_label = QLabel("Erkannte Tumorzellen (KI): 0\nKorrekte Anzahl: 0", self.rightContainer)
        self.model_annot_label.setFont(bold_font)
        
        self.SummaryLayout.addItem(QSpacerItem(80, 80))
        self.SummaryLayout.addWidget(self.user_time_label, alignment=Qt.AlignLeft)
        self.SummaryLayout.addWidget(self.user_annot_label, alignment=Qt.AlignLeft)
        
        self.SummaryLayout.addItem(QSpacerItem(80, 80))
        self.SummaryLayout.addWidget(self.model_time_label, alignment=Qt.AlignLeft)
        self.SummaryLayout.addWidget(self.model_annot_label, alignment=Qt.AlignLeft)

        self.SummaryLayout.addItem(QSpacerItem(80, 80, QSizePolicy.Minimum, QSizePolicy.Expanding))

        rightVerticalLayout.addLayout(self.SummaryLayout)  # ✅ Add Summary Row

        self.rightContainer.setLayout(rightVerticalLayout)

        self.display_image()
        
        palette = self.palette()
        palette.setColor(self.backgroundRole(), Qt.black)
        palette.setColor(self.foregroundRole(), Qt.white)
        self.setPalette(palette)
        #self.showMaximized()
        
    
    def create_image(self):
        
        self.rectangles = []
        
        curr_size = self.image_size.text()
        
        if curr_size:
            print(curr_size)
            if ',' in curr_size:
                x, y = curr_size.split(',')
                x, y = int(x), int(y)
            else:
                x, y = int(curr_size), int(curr_size)

        else:
            x = y = 512
            
        image, mask_image, idxs, positions, target = self.generate_image((x, y), self.cells, self.masks, self.targets, 1000, 250)
        
        self.ratio = int(0.8*(2*WIDTH//3))/x
        
        self.curr_targets = target
        self.positions = np.array(positions) * self.ratio        
        self.curr_image = resize_with_scipy(image * 255, int(0.8*(2*WIDTH//3)), int(0.8*(2*WIDTH//3)))
        self.orig_curr_image = deepcopy(self.curr_image)
        self.curr_mask_image = resize_with_scipy(mask_image * 255, int(0.8*(2*WIDTH//3)), int(0.8*(2*WIDTH//3))) 

        self.curr_pixmap = qimage2ndarray.array2qimage(self.curr_image)
        #self.curr_pixmap = qimage2ndarray.array2qimage(np.stack([self.curr_mask_image * 0.5]*3).transpose(1,2,0) + self.curr_image)
        self.Image.setPixmap(QtGui.QPixmap.fromImage(self.curr_pixmap))
                
    def display_image(self):
        """ Generates a NumPy image and displays it in ClickableLabel """
    
        self.curr_pixmap = qimage2ndarray.array2qimage(np.zeros((int(0.8*(2*WIDTH//3)), int(0.8*(2*WIDTH//3)))))
        self.Image.setPixmap(QtGui.QPixmap.fromImage(self.curr_pixmap))
    

    def create_button(self, text, min_width, max_width, max_height, alignment):
        button = QPushButton(text, self.rightContainer)
        button.setMinimumSize(min_width, 0)
        button.setMaximumSize(max_width, max_height)
        self.rightContainer.layout().addWidget(button, 0, alignment)
        return button

    def create_label(self, text, max_width, max_height, alignment):
        label = QLabel(text, self.rightContainer)
        label.setMaximumSize(max_width, max_height)
        self.rightContainer.layout().addWidget(label, 0, alignment)
        return label

    def connect_signals(self):
        self.new_image_button.clicked.connect(self.create_image)
        self.Image.imageClicked.connect(self.image_clicked)
        self.start.clicked.connect(self.start_timer)
        self.stop.clicked.connect(self.stop_timer)
        self.show_results.clicked.connect(self.show_model_predictions)

    def run_model(self):
        

        self.model_timer = time()
    
        ret = run_model(self.model, self.dataloader, n_samples=10_000)

        # Convert embeddings to tensor
        X = torch.tensor(np.array(ret['embeddings'].tolist()).mean(axis=(2,3)))
        print(X.shape)

        # Get predictions
        pred = self.classifier(X)  # Predicted probabilities
        self.predictions  = (pred > 0.5).int().cpu().numpy().flatten()  # Convert to binary labels (0 or 1)
        
        self.model_timer = time() - self.model_timer
        
        self.person_predictions = np.zeros_like(self.predictions)


    def start_timer(self):
    
        if self.started:
            return 
        
        self.timer = time()
        self.started = True
        
        self.user_time_label.setText(f"Deine Zeit: ---")
        self.user_annot_label.setText(f"Erkannte Tumorzellen: ---\nKorrekte Anzahl: ---")

        self.model_time_label.setText(f"Zeit der KI: ---")
        self.model_annot_label.setText(f"Erkannte Tumorzellen (KI): ---\nKorrekte Anzahl: ---")
        # self.user_time_label = QLabel("User Time: 0s", self.rightContainer)
        # self.user_annot_label = QLabel("User: 0\nCorrect: 0", self.rightContainer)
        
        # self.model_time_label = QLabel("Model Time: 0s", self.rightContainer)
        # self.model_annot_label = QLabel("Model: 0\nCorrect: 0", self.rightContainer)
        
    def stop_timer(self):
        """ Stops the timer and updates the summary row with user and model statistics """
        
        self.timer = time() - self.timer
        self.started = False
    
        # Compute user-annotated vs. correct
        user_annotations = np.sum(self.person_predictions)  # How many the user annotated
        correct_user = np.sum(self.person_predictions == self.curr_targets) / len(self.curr_targets) # How many were correct

        # Compute model-annotated vs. correct
        model_annotations = np.sum(self.predictions)  # How many the model annotated
        correct_model = np.sum(self.predictions == self.curr_targets) / len(self.curr_targets) # How many were correct

        # Update labels with the results
        self.user_time_label.setText(f"Deine Zeit: {self.timer:.2f}s")
        self.user_annot_label.setText(f"Erkannte Tumorzellen: {user_annotations}\nKorrekte Anzahl: {correct_user:.0%}")

        self.model_time_label.setText(f"Zeit der KI: {self.model_timer:.2f}s")
        self.model_annot_label.setText(f"Erkannte Tumorzellen (KI): {model_annotations}\nKorrekte Anzahl: {correct_model:.0%}")

        print(f"User Time: {self.timer:.2f}s")
        print(f"User Annotations: {user_annotations}, Correct: {correct_user}")
        print(f"Model Time: {self.model_timer:.2f}s")
        print(f"Model Annotations: {model_annotations}, Correct: {correct_model}")
        
    def show_model_predictions(self):
        
        self.rectangles = []
        for (pos_y, pos_x), user_prediction, model_prediction in zip(self.positions, self.person_predictions, self.predictions):
            
            same = 0 if user_prediction == model_prediction else 2
            self.drawRectanglesAndText(pos_y-(self.ratio*96)//2, pos_x-(self.ratio*96)//2, self.ratio*96, self.ratio*96, same, '')

        
    def image_clicked(self, pos):
        
        if not self.started:
            return
        if self.curr_mask_image[pos.y(), pos.x()] == 0:
            return
        else:
            min_idx = np.argmin(cdist(self.positions, np.atleast_2d(np.array([pos.y(), pos.x()]))))
            
            pos_y, pos_x = self.positions[min_idx]
            
            self.drawRectanglesAndText(pos_y-(self.ratio*96)//2, pos_x-(self.ratio*96)//2, self.ratio*96, self.ratio*96, 1, '')
            
            self.person_predictions[min_idx] = 1
            

    def drawRectanglesAndText(self, r_min, c_min, r_size, c_size, prediction, uncertainty):
        # Convert NumPy array to QImage
        qimage = qimage2ndarray.array2qimage(self.curr_image)  # Convert np array to QImage
        pixmap = QPixmap.fromImage(qimage)  # Convert QImage to QPixmap

        painter = QPainter(pixmap)
        font = QFont("Arial", 20)
        painter.setFont(font)

        # Define color based on prediction
        if prediction == 0:
            color = QColor(0, 255, 0)  # Green
        elif prediction == 1:
            color = QColor(255, 255, 0)  # Yellow
        elif prediction == 2:
            color = QColor(255, 0, 0)  # Red
        else:
            print(prediction)
            return  # Exit function if invalid prediction

        # Define the rectangle
        rect = QRect(c_min, r_min, c_size, r_size)
        self.rectangles.append([rect, color])

        # Draw all rectangles stored
        for rectangle in self.rectangles:
            painter.setPen(QPen(rectangle[1], 2))
            painter.drawRect(rectangle[0])

        # Draw the text
        painter.setPen(QColor(0, 255, 0))  # Green text
        text_position = QPoint(c_min, r_min - 5)
        painter.drawText(text_position, str(uncertainty))  # Convert uncertainty to string

        painter.end()

        # Set the new pixmap with drawn elements
        self.Image.setPixmap(pixmap)
        
    

if __name__ == '__main__':

    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())
