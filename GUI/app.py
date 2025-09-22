import sys

sys.path.append(".")

from pathlib import Path
from time import time

import numpy as np
from copy import deepcopy
from time import time
import torch
import h5py
import tifffile
import qimage2ndarray
from scipy.ndimage import zoom
from scipy.spatial.distance import cdist
from cellpose import models, io
#from einops import rearrange
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import Qt, QRect, pyqtSignal, QPoint, QPointF
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QFont, QImage
from PyQt5.QtWidgets import QApplication, QMainWindow, QSlider, QCheckBox, QComboBox, QSpacerItem, QSizePolicy, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QTableWidget, QTableWidgetItem
from PyQt5.uic import loadUi  # Import to load .ui files
from PyQt5.QtWidgets import QApplication
import sys
import pandas as pd

import matplotlib.pyplot as plt
from cellplot.segmentation import rand_col_seg, contoure_seg
from cellplot.patches import gridPlot

from mmengine.registry import build_model_from_cfg
from mmselfsup.registry import MODELS
from mmselfsup.utils import register_all_modules
from mmengine.runner import Runner
from mmengine.config import Config

# your custom modules
from src_ import losses, mvsimclr, dataset, transforms
import torch

STYLESHEET_CLICKED = """QPushButton {
                background-color: green;
                color: white;
                border: 1px solid white;
                padding: 5px;
                border-radius: 5px;
            }"""

STYLESHEET_NOT_CLICKED = """QPushButton {
                background-color: #555;
                color: white;
                border: 1px solid white;
                padding: 5px;
                border-radius: 5px;
            }"""

MAX_ON = 2
STYLED = (STYLESHEET_CLICKED, STYLESHEET_NOT_CLICKED)

register_all_modules()

from CellPatchExtraction.CellPatchExtraction.src.extraction import segment_image, extract_and_pad_objects

from PyQt5.QtGui import QFont

from pathlib import Path
BASEDIR = Path(__file__).parent.parent

# Define a font (Font Name, Font Size)
bold_font = QFont("Arial", 20)
bold_font.setBold(True)  # Make text bold

CLASS_MAP = {0: 0, 1:2, 3:1, 2:3}

from src import randomly_place_cells, load_model

from screeninfo import get_monitors

WIDTH, HEIGHT = get_monitors()[0].width, get_monitors()[0].height
if (WIDTH / HEIGHT) > 16/9:
    print('resorting to 1920')
    WIDTH = 1920
    HEIGHT = 1056

H5_PATH = BASEDIR.joinpath('small_data.h5')
REAL_IMAGE_PATH = BASEDIR.joinpath('real_images')

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



class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.setup_layout()
        self.connect_signals()
        self.cellpose_model = self.get_cellpose_model()

        print(self.cellpose_model)
        
        self.load_h5(H5_PATH)
        self.model = self.get_model()

        self.ANNOTATION_MODE = False
        self.MODEL_RAN = False

        self.curr_display = None

        self.GT_CLICKED = False
        self.AI_CLICKED = False
        self.USER_CLICKED = False
        
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

    @staticmethod
    def get_cellpose_model():

        model = models.CellposeModel(
            gpu=False, device=torch.device('mps'),
            pretrained_model='/Users/simon.gutwein/src/TEDxAI_2025/CP_TU_MORE'
        )

        if torch.backends.mps.is_available():
            model.device = torch.device('mps')
            model.net = model.net.to(device='mps')

        elif torch.cuda.is_available():
            model.device = torch.device('cuda')
            model.net = model.net.to(device='cuda')
    
        return model
    
    def get_model(self):

        cfg = Config.fromfile('/Users/simon.gutwein/src/TEDxAI_2025/config_new.py').to_dict()
        model = build_model_from_cfg(cfg['model'], MODELS)
        checkpoint = torch.load('/Users/simon.gutwein/src/TEDxAI_2025/model_new.pth', map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        model = model.eval().to('mps')

        return model
    
        
    def load_h5(self, path):

        with h5py.File(path, 'r') as f:
            
            self.patches = f['patches'][()]
            self.masks = f['masks'][()]
            self.classes = f['classes'][()]
            

    def setup_layout(self):

        self.setWindowTitle("MainWindow")
        self.centralWidget = QWidget(self)
        self.setCentralWidget(self.centralWidget)

        self.mainHorizontalLayout = QHBoxLayout(self.centralWidget)
        self.mainHorizontalLayout.setAlignment(Qt.AlignCenter)

        self.leftContainer = QWidget(self.centralWidget)
        self.leftContainer.setMinimumWidth(int(2*WIDTH//3)-10)

        self.rightContainer = QWidget(self.centralWidget)
        self.rightContainer.setMinimumWidth(int(WIDTH//3)-10)
        self.rightContainer.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

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
        rightVerticalLayout.setSpacing(10)

        # HBox for Image Size and Load Image Button
        self.synthetic_image = QHBoxLayout()
        self.image_size = QLineEdit(self)
        self.image_size.setMinimumSize(250, 35)
        self.image_size.setMaximumSize(250, 35)
        self.image_size.setPlaceholderText("Synthetic Image Size...")
        self.synthetic_image.addWidget(self.image_size)
        self.synthetic_image_button = self.create_button("Bild Generieren", 200, 250, 35, Qt.AlignHCenter)
        self.synthetic_image.addWidget(self.synthetic_image_button)


        self.real_image = QHBoxLayout()
        self.image_path = QComboBox(self)
        self.image_path.setMinimumSize(250, 35)
        self.image_path.setMaximumSize(250, 35)

        image_files = list(REAL_IMAGE_PATH.glob('*'))
        self.image_files = {path.stem: path for path in image_files}

        self.image_path.addItems(list(self.image_files.keys()))
        self.image_path.setEditable(True)
        self.image_path.setInsertPolicy(QComboBox.NoInsert)
        self.image_path.setPlaceholderText("Choose Image")
        self.real_image.addWidget(self.image_path)
        self.real_image_button = self.create_button("Bild Laden", 200, 250, 35, Qt.AlignHCenter)
        self.real_image.addWidget(self.real_image_button)

        startStopLayout = QHBoxLayout()
        self.start = self.create_button("Suche beginnen...", 100, 250, 120, Qt.AlignHCenter)
        self.start.setMinimumSize(120, 40)  # Set minimum width and height
        self.start.setFont(bold_font)
        self.start.setStyleSheet("background-color: green; color: white; border-radius: 10px; font-weight: bold;")
        self.stop = self.create_button("Suche beenden...", 100, 250, 120, Qt.AlignHCenter)
        self.stop.setMinimumSize(120, 40)  # Set minimum width and height
        self.stop.setFont(bold_font)
        self.stop.setStyleSheet("background-color: red; color: white; border-radius: 10px; font-weight: bold;")
        startStopLayout.addWidget(self.start)
        startStopLayout.addWidget(self.stop)
        
        showResults = QHBoxLayout()
        self.ai_results = self.create_button("AI CLASSFICATION", 35, 550, 35, Qt.AlignHCenter)
        self.ai_results.setMinimumSize(120, 40)  # Set minimum width and height
        self.gt_results = self.create_button("SHOW GROUND TRUTH", 35, 550, 35, Qt.AlignHCenter)
        self.gt_results.setMinimumSize(120, 40)  # Set minimum width and height
        self.user_results = self.create_button("USER ANNOTATIONS", 35, 550, 35, Qt.AlignHCenter)
        self.user_results.setMinimumSize(120, 40)  # Set minimum width and height
        showResults.addWidget(self.ai_results)
        showResults.addWidget(self.user_results)
        showResults.addWidget(self.gt_results)

        filter_layout = QHBoxLayout()
        self.filter_label = QLabel("Filter: 0.0", self)
        self.filter_label.setAlignment(Qt.AlignCenter)
        self.filter_toggle = QCheckBox("Filter Low")

        filter_layout.addWidget(self.filter_label)
        filter_layout.addWidget(self.filter_toggle)

        self.filter_slider = QSlider(Qt.Horizontal, self)
        self.filter_slider.setRange(0, 100)
        self.filter_slider.setValue(100)
        self.filter_slider.setTickPosition(QSlider.TicksBelow)
        self.filter_slider.setTickInterval(10)
        self.filter_slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        slider_layout = QVBoxLayout()
        slider_layout.addItem(QSpacerItem(0, 20))
        slider_layout.addLayout(filter_layout)
        slider_layout.addWidget(self.filter_slider)

        self.model_timer_label = QLabel("", self)
        self.model_timer_label.setAlignment(Qt.AlignCenter)

        self.user_timer_label = QLabel("", self)
        self.user_timer_label.setAlignment(Qt.AlignCenter)

        self.model_timer_label.setStyleSheet("font-size: 14pt;")
        self.user_timer_label.setStyleSheet("font-size: 14pt;")

        slider_layout.addItem(QSpacerItem(0, 20))
        slider_layout.addWidget(self.user_timer_label)
        slider_layout.addWidget(self.model_timer_label)

        cell_image = QVBoxLayout(self.rightContainer)
        self.cell_image = ClickableLabel(self.rightContainer)
        self.cell_image.setMinimumSize(384, 384)
        self.cell_image.setAlignment(Qt.AlignCenter)
        cell_image.addWidget(self.cell_image)

        rightVerticalLayout.addLayout(cell_image)
        rightVerticalLayout.addLayout(self.synthetic_image)
        rightVerticalLayout.addLayout(self.real_image)
        rightVerticalLayout.addLayout(startStopLayout)
        rightVerticalLayout.addLayout(showResults)
        rightVerticalLayout.addLayout(slider_layout)

        self.make_black_image()
        
        palette = self.palette()
        palette.setColor(self.backgroundRole(), Qt.black)
        palette.setColor(self.foregroundRole(), Qt.white)
        self.setPalette(palette)
        

    
    def create_image(self):

        self.reset_clicked()

        self.ANNOTATION_MODE = False
        self.MODEL_RAN = False
        self.GT_AVAILABLE = True

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
            
        self.curr_image, self.curr_masks, idxs, self.curr_coords, self.curr_targets, self.curr_patches = randomly_place_cells(
            (x,y), 
            self.patches, 
            self.masks, 
            self.classes, 
            1000,
            max_rejections=100
        )

        self.curr_patches = np.array(self.curr_patches)
        print(self.curr_patches.shape)
        self.curr_targets = np.array(self.curr_targets)

        W, H, _ = self.curr_image.shape
        target = int(0.8 * (2 * WIDTH // 3))

        scale = target / max(W, H)
        new_W, new_H = int(W * scale), int(H * scale)

        self.show_image = resize_with_scipy(self.curr_image, new_W, new_H).astype(np.float32)
        self.show_image /= self.show_image.max()
        self.curr_image = self.curr_image / self.curr_image.max()

        mask_outlines = contoure_seg(self.curr_masks, ret_rgb=True)
        mask_outlines = resize_with_scipy(mask_outlines, new_W, new_H).astype(np.float32)

        self.curr_display = np.clip(self.show_image + 0.5*mask_outlines, 0, 1) * 255

        self.curr_pixmap = qimage2ndarray.array2qimage(self.curr_display)
        self.Image.setPixmap(QtGui.QPixmap.fromImage(self.curr_pixmap))

        self.curr_coords = np.array(self.curr_coords) * scale

        self.user_clicked_dict = pd.DataFrame(dict(
            patches=list(self.curr_patches), 
            coords=list(self.curr_coords), 
            ground_truth=[CLASS_MAP[c] for c in self.curr_targets],
            user=[0]*len(self.curr_coords)))
        
        self.curr_masks = resize_with_scipy(self.curr_masks, new_W, new_H)
                
    def make_black_image(self):
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
    
    def load_image(self):

        self.reset_clicked()

        self.ANNOTATION_MODE = False
        self.MODEL_RAN = False
        self.GT_AVAILABLE = False

        image_path = self.image_files[self.image_path.currentText()]
        self.curr_image = tifffile.imread(image_path)
        
        W, H, _ = self.curr_image.shape
        target = int(0.8 * (2 * WIDTH // 3))

        scale = target / max(W, H)
        new_W, new_H = int(W * scale), int(H * scale)

        self.show_image = resize_with_scipy(self.curr_image, new_W, new_H).astype(np.float32)
        self.show_image /= self.show_image.max()
        self.curr_image = self.curr_image / self.curr_image.max()
        self.curr_masks, _ = segment_image(self.curr_image[..., 2], model=self.cellpose_model, cellpose_kwargs=dict(diameter=35))

        mask_outlines = contoure_seg(self.curr_masks, ret_rgb=True)
        mask_outlines = resize_with_scipy(mask_outlines, new_W, new_H).astype(np.float32)

        self.curr_display = np.clip(self.show_image + 0.5*mask_outlines, 0, 1) * 255

        self.curr_pixmap = qimage2ndarray.array2qimage(self.curr_display)
        self.Image.setPixmap(QtGui.QPixmap.fromImage(self.curr_pixmap))

        self.curr_patches, _, _, _, self.curr_coords = extract_and_pad_objects(
            self.curr_masks, 
            self.curr_image, 
            128, 
            exclude_edges=False, 
            use_surrounding=False, 
            dilate_mask=2
        )

        self.curr_coords = np.array(self.curr_coords) * scale

        self.user_clicked_dict = pd.DataFrame(dict(
            patches=list(self.curr_patches), 
            coords=list(self.curr_coords), 
            user=[0]*len(self.curr_coords)))
        
        self.curr_masks = resize_with_scipy(self.curr_masks, new_W, new_H)


    def update_uncertainty_filter(self):

        slider_value = self.filter_slider.value() / 100

        if not self.AI_CLICKED or not self.MODEL_RAN: 
            return
        
        self.drawRectanglesAndText(mode='ai', uncertainty=slider_value)


    def connect_signals(self):

        self.synthetic_image_button.clicked.connect(self.create_image)
        self.Image.imageClicked.connect(self.image_clicked)
        self.start.clicked.connect(self.start_timer)
        self.stop.clicked.connect(self.stop_timer)
        self.real_image_button.clicked.connect(self.load_image)
        self.filter_slider.valueChanged.connect(self.update_filter_value)
        self.filter_slider.sliderReleased.connect(self.update_uncertainty_filter)
        self.filter_toggle.toggled.connect(self.update_uncertainty_filter)
        self.gt_results.clicked.connect(self.gt_clicked)
        self.ai_results.clicked.connect(self.ai_clicked)
        self.user_results.clicked.connect(self.user_clicked)
        self.update_filter_value(100)

    def _count_on(self):
        return int(self.GT_CLICKED) + int(self.AI_CLICKED) + int(self.USER_CLICKED)

    def _try_toggle(self, attr_name, widget):
        cur = getattr(self, attr_name)
        # allow turning off always; block turning on if already 2 on
        if not cur and self._count_on() >= MAX_ON:
            return  # ignore this click

        new = not cur
        setattr(self, attr_name, new)
        widget.setStyleSheet(STYLED[0] if new else STYLED[1])
        self.switch_show()

    def gt_clicked(self):
        self._try_toggle("GT_CLICKED", self.gt_results)

    def ai_clicked(self):

        if not self.MODEL_RAN:
            self.run_model()

        self._try_toggle("AI_CLICKED", self.ai_results)

    def user_clicked(self):
        self._try_toggle("USER_CLICKED", self.user_results)

    def switch_show(self):

        n_toggled = (self.GT_CLICKED + self.AI_CLICKED + self.USER_CLICKED)

        if n_toggled == 0:
            self.plot_only_image()

        elif n_toggled == 1:
            self.plot_single_annotations()

        elif n_toggled == 2:
            self.plot_comparison()
        else: 
            print('SHOULD NEVER HAPPEN')

    def plot_only_image(self):

        if self.curr_display is not None:
            qimage = qimage2ndarray.array2qimage(self.curr_display)  # Convert np array to QImage
            pixmap = QPixmap.fromImage(qimage)  # Convert QImage to QPixmap
            self.Image.setPixmap(pixmap)

    def plot_single_annotations(self):

        if self.GT_CLICKED:
            self.drawRectanglesAndText(mode='ground_truth')
        elif self.USER_CLICKED:
            self.drawRectanglesAndText(mode='user')
        elif self.AI_CLICKED:
            self.drawRectanglesAndText(mode='ai')

    def plot_comparison(self):

        if self.GT_CLICKED and self.USER_CLICKED:
            self.drawRectanglesAndText(mode='comparison', comp1='ground_truth', comp2='user')

        if self.GT_CLICKED and self.AI_CLICKED:
            self.drawRectanglesAndText(mode='comparison', comp1='ground_truth', comp2='ai')

        if self.AI_CLICKED and self.USER_CLICKED:
            self.drawRectanglesAndText(mode='comparison', comp1='ai', comp2='user')


    def reset_clicked(self):

        self.GT_CLICKED = self.USER_CLICKED = self.AI_CLICKED = False
        for widget in [self.gt_results, self.user_results, self.ai_results]:
            widget.setStyleSheet(STYLED[1])


    def update_filter_value(self, value):
        """Update label with mapped slider value (0–1)."""
        self.filter_label.setText(f"Filter: {value}%")
        # TODO: Apply filter logic to your data here

    def run_model(self):

        self.model_timer = time()

        results = self.model.predict([torch.tensor(np.array(self.curr_patches).astype(np.float32).transpose(0,3,1,2))], return_uncertainty=True)

        self.model_timer = time() - self.model_timer

        self.user_clicked_dict['ai'] = [CLASS_MAP[c] for c in results['classification']]
        self.user_clicked_dict['regression'] = results['regression']
        self.user_clicked_dict['uncertainty'] = results['uncertainty']

        self.model_timer_label.setText(f"Model Prediction Time - {self.model_timer:.2f}s - {(self.model_timer/len(self.user_clicked_dict))*1000:.2f}ms per cell")

        self.MODEL_RAN = True


    def start_timer(self):
    
        if self.ANNOTATION_MODE:
            return 
        self.ANNOTATION_MODE = True
        
        self.timer = time()
        
    def stop_timer(self):
        """ Stops the timer and updates the summary row with user and model statistics """
        
        if not self.ANNOTATION_MODE:
            return
        self.ANNOTATION_MODE = False

        self.timer = time() - self.timer
        self.user_timer_label.setText(f"User Time - {self.timer:.2f}s - {(self.timer/len(self.user_clicked_dict))*1000:.2f}ms per cell")
  
    def image_clicked(self, pos):

        self.only_show = True
        if (self.AI_CLICKED + self.USER_CLICKED + self.GT_CLICKED) == 0 and self.ANNOTATION_MODE:
            self.only_show = False
        
        if self.curr_masks[pos.y(), pos.x()] == 0:
            return
    
        min_idx = np.argmin(cdist(self.curr_coords, np.atleast_2d(np.array([pos.x(), pos.y()]))))

        if not self.only_show:
            if self.user_clicked_dict['user'].iloc[min_idx] == 3:
                self.user_clicked_dict['user'].iloc[min_idx] = 0
            else:
                self.user_clicked_dict['user'].iloc[min_idx] += 1

        self.tmp = qimage2ndarray.array2qimage(resize_with_scipy(self.user_clicked_dict['patches'].iloc[min_idx]*255, 384, 384))
        self.cell_image.setPixmap(QtGui.QPixmap.fromImage(self.tmp))
        
        if not self.ANNOTATION_MODE:
            self.switch_show()
        else:
            self.drawRectanglesAndText(mode='user')
    
            

    def drawRectanglesAndText(self, mode, uncertainty=None, comp1=None, comp2=None):

        _user_clicked_dict = self.user_clicked_dict.copy()  # rename to df if it’s really a DataFrame
        
        if uncertainty is not None:
            if self.filter_toggle.isChecked():
                _user_clicked_dict = _user_clicked_dict[(1-_user_clicked_dict['uncertainty']) > uncertainty]
            else:
                _user_clicked_dict = _user_clicked_dict[(1-_user_clicked_dict['uncertainty']) < uncertainty]

        qimage = qimage2ndarray.array2qimage(self.curr_display)  # Convert np array to QImage
        pixmap = QPixmap.fromImage(qimage)  # Convert QImage to QPixmap

        painter = QPainter(pixmap)
        font = QFont("Arial", 10)
        painter.setFont(font)

        colors_dict = {
            0: {'color': QColor(64 ,64, 64), 'class': 'NORMAL'},
            1: {'color': QColor(255, 255, 0), 'class': 'LOSS'},
            2: {'color': QColor(0, 255, 255), 'class': 'GAIN'},
            3: {'color': QColor(255, 0, 255), 'class': 'AMP'}, 
            4: {'color': QColor(255, 0, 0), 'class': ''},
            5: {'color': QColor(0, 255, 0), 'class': ''}  
        }

        if mode=='ground_truth' and not self.GT_AVAILABLE:
            painter.end()
            self.Image.setPixmap(pixmap)
            return 

        elif mode == 'comparison':

            if not self.GT_AVAILABLE and self.GT_CLICKED:
                painter.end()
                self.Image.setPixmap(pixmap)
                return 

            for _, row in _user_clicked_dict.iterrows():

                if row[comp1] == row[comp2]:
                    color = colors_dict[5]['color']
                    class_text = colors_dict[5]['class']
                else:
                    color = colors_dict[4]['color']
                    class_text = colors_dict[4]['class']

                rectangle = QRect(row.coords[0]-32, row.coords[1]-32, 64, 64)
                painter.setPen(QPen(color, 2))
                painter.drawRect(rectangle)

                text = class_text
                spacer = 0

                painter.setPen(QColor(255, 255, 255))
                fm = painter.fontMetrics()
                text_rect = fm.boundingRect(QRect(0, 0, 0, 0), Qt.AlignCenter | Qt.TextWordWrap, text)
                text_position = QPoint(row.coords[0], row.coords[1]-42+spacer)
                text_rect.moveCenter(text_position)
                painter.drawText(text_rect, Qt.AlignCenter | Qt.TextWordWrap, text)
            
        else:

            for _, row in _user_clicked_dict.iterrows():
                
                if row.user == 0 and mode == 'user':
                    continue
                
                if mode == 'user':
                    color = colors_dict[row.user]['color']
                    class_text = colors_dict[row.user]['class']

                elif mode == 'ai':
                    color = colors_dict[row.ai]['color']
                    class_text = colors_dict[row.ai]['class']

                elif mode == 'ground_truth':
                    print(row.ground_truth)
                    print(colors_dict[row.ground_truth])
                    print(colors_dict[row.ground_truth]['color'])
                    color = colors_dict[row.ground_truth]['color']
                    class_text = colors_dict[row.ground_truth]['class']

                rectangle = QRect(row.coords[0]-32, row.coords[1]-32, 64, 64)
                painter.setPen(QPen(color, 2))
                painter.drawRect(rectangle)

                if mode == 'user' or mode == 'ground_truth':
                    text = class_text
                    spacer = 0
                else: 
                    text = f'{1 - row["uncertainty"]:.0%}\n{class_text}'
                    spacer = - 4

                painter.setPen(QColor(255, 255, 255))
                fm = painter.fontMetrics()
                text_rect = fm.boundingRect(QRect(0, 0, 0, 0), Qt.AlignCenter | Qt.TextWordWrap, text)
                text_position = QPoint(row.coords[0], row.coords[1]-42+spacer)
                text_rect.moveCenter(text_position)
                painter.drawText(text_rect, Qt.AlignCenter | Qt.TextWordWrap, text)


        painter.end()

        # Set the new pixmap with drawn elements
        self.Image.setPixmap(pixmap)
        
    

if __name__ == '__main__':

    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())
