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
from PyQt5.QtCore import Qt, QRect, pyqtSignal, QPoint, QPointF, QThread, QTimer
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QFont, QImage
from PyQt5.QtWidgets import QApplication, QMainWindow, QSlider, QCheckBox, QComboBox, QSpacerItem, QSizePolicy, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QTableWidget, QTableWidgetItem, QProgressBar
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

HELP_UNCERTAINTY_THRESHOLD = 0.4  # uncertainty > this = AI wants a second opinion

HELP_STYLESHEET_OFF = """QPushButton {
                background-color: #3a1f1f;
                color: #ff8080;
                border: 2px dashed #ff5050;
                padding: 5px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #552727; }"""

HELP_STYLESHEET_ON = """QPushButton {
                background-color: #cc3030;
                color: white;
                border: 2px solid #ff8080;
                padding: 5px;
                border-radius: 5px;
                font-weight: bold;
            }"""


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


register_all_modules()

from CellPatchExtraction.src.extraction import segment_image, extract_and_pad_objects

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
CELLPOSE_MODEL_PATH = str(BASEDIR.joinpath('CP_TU_MORE'))
MODEL_PATH = BASEDIR.joinpath('model_new.pth')
CONFIG_PATH = BASEDIR.joinpath('config_new.py')

def resize_with_scipy(image, target_height, target_width):
    """Resize an image using scipy to a target height and width."""
    scale = (target_height / image.shape[0], target_width / image.shape[1])
    scale = scale if image.ndim == 2 else scale + (1,)
    return zoom(image, scale, order=0)

class InferenceWorker(QThread):

    finished_ok = pyqtSignal(dict, float)
    failed = pyqtSignal(str)
    progress = pyqtSignal(int, int)  # done, total

    CHUNK_SIZE = 8
    MIN_CHUNK_DELAY = 0.05  # seconds; ensures the progress bar is visible even on fast GPUs

    def __init__(self, model, patches, parent=None):
        super().__init__(parent)
        self.model = model
        self.patches = patches

    def run(self):
        try:
            t0 = time()
            tensor = torch.tensor(self.patches.astype(np.float32).transpose(0, 3, 1, 2))
            total = tensor.shape[0]
            self.progress.emit(0, total)

            chunks = {'classification': [], 'regression': [], 'uncertainty': []}

            done = 0
            with torch.no_grad():
                for start in range(0, total, self.CHUNK_SIZE):
                    chunk_start = time()
                    end = min(start + self.CHUNK_SIZE, total)
                    sub = tensor[start:end]
                    r = self.model.predict([sub], return_uncertainty=True)
                    for k in chunks:
                        chunks[k].append(np.asarray(r[k]))
                    done = end
                    self.progress.emit(done, total)
                    elapsed = time() - chunk_start
                    if elapsed < self.MIN_CHUNK_DELAY:
                        self.msleep(int((self.MIN_CHUNK_DELAY - elapsed) * 1000))

            results = {k: np.concatenate(v, axis=0) for k, v in chunks.items()}
            self.finished_ok.emit(results, time() - t0)
        except Exception as e:
            self.failed.emit(repr(e))


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

        self.device = pick_device()

        self.setup_layout()
        self.connect_signals()
        self.cellpose_model = self.get_cellpose_model()

        self.load_h5(H5_PATH)
        self.model = self.get_model()

        self.ANNOTATION_MODE = False
        self.MODEL_RAN = False
        self.ANNOTATED = False
        self.GT_AVAILABLE = False

        self.curr_display = None
        self.curr_masks = None
        self.curr_patches = None
        self.user_clicked_dict = None

        self.GT_CLICKED = False
        self.AI_CLICKED = False
        self.USER_CLICKED = False
        self.HELP_CLICKED = False

        self._inference_worker = None
        self._inference_running = False
        self._ai_toggle_pending = False
        self._help_toggle_pending = False
        self._inference_patches_id = None

        self._ai_reveal_timer = QTimer(self)
        self._ai_reveal_timer.setInterval(30)
        self._ai_reveal_timer.timeout.connect(self._ai_reveal_tick)
        self._ai_revealed_order = None
        self._ai_revealed_count = 0
        self._ai_revealed_total = 0
        self._ai_animation_active = False

        self._score_timer = QTimer(self)
        self._score_timer.setInterval(25)
        self._score_timer.timeout.connect(self._score_tick)
        self._score_target = None  # (tag, matches, total, current_step)
        self._score_steps_total = 30
        
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

    def get_cellpose_model(self):

        model = models.CellposeModel(
            gpu=False, device=self.device,
            pretrained_model=CELLPOSE_MODEL_PATH
        )
        model.device = self.device
        model.net = model.net.to(device=self.device)
        return model

    def get_model(self):

        cfg = Config.fromfile(CONFIG_PATH).to_dict()
        model = build_model_from_cfg(cfg['model'], MODELS)
        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        return model.eval().to(self.device)
    
        
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
        self.synthetic_image_button = self.create_button("Generate Image", 200, 250, 35, Qt.AlignHCenter)
        self.synthetic_image.addWidget(self.synthetic_image_button)


        self.real_image = QHBoxLayout()
        self.image_path = QComboBox(self)
        self.image_path.setMinimumSize(250, 35)
        self.image_path.setMaximumSize(250, 35)

        image_files = [f for f in list(REAL_IMAGE_PATH.glob('*')) if 'tif' in str(f).lower()]
        self.image_files = {path.stem: path for path in image_files}

        self.image_path.addItems(list(self.image_files.keys()))
        self.image_path.setEditable(True)
        self.image_path.setInsertPolicy(QComboBox.NoInsert)
        self.image_path.setPlaceholderText("Choose Image")
        self.real_image.addWidget(self.image_path)
        self.real_image_button = self.create_button("Load Image", 200, 250, 35, Qt.AlignHCenter)
        self.real_image.addWidget(self.real_image_button)

        startStopLayout = QHBoxLayout()
        self.start = self.create_button("Start Annotation", 100, 250, 120, Qt.AlignHCenter)
        self.start.setMinimumSize(120, 40)  # Set minimum width and height
        self.start.setFont(bold_font)
        self.start.setStyleSheet("background-color: green; color: white; border-radius: 10px; font-weight: bold;")
        self.stop = self.create_button("End Annotation", 100, 250, 120, Qt.AlignHCenter)
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

        helpLayout = QHBoxLayout()
        self.help_results = self.create_button("AI NEEDS HELP (low-confidence cells)", 35, 800, 40, Qt.AlignHCenter)
        self.help_results.setMinimumSize(200, 40)
        self.help_results.setStyleSheet(HELP_STYLESHEET_OFF)
        helpLayout.addWidget(self.help_results)

        slider_layout = QVBoxLayout()
        slider_layout.addItem(QSpacerItem(0, 20))

        self.timer_text = ""
        self.timer_label = QLabel(self.timer_text, self)
        self.timer_label.setAlignment(Qt.AlignCenter)
        self.timer_label.setStyleSheet("font-size: 14pt;")

        self.score_label = QLabel("", self)
        self.score_label.setAlignment(Qt.AlignCenter)
        self.score_label.setStyleSheet("font-size: 18pt; font-weight: bold; color: #7CFC00;")

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Classifying cell %v / %m")
        self.progress_bar.setStyleSheet(
            """
            QProgressBar {
                border: 1px solid #888;
                border-radius: 6px;
                background-color: #222;
                color: white;
                text-align: center;
                font-size: 12pt;
                min-height: 28px;
            }
            QProgressBar::chunk {
                background-color: #00B4FF;
                border-radius: 5px;
            }
            """
        )
        self.progress_bar.setVisible(False)

        slider_layout.addItem(QSpacerItem(0, 20))
        slider_layout.addWidget(self.timer_label)
        slider_layout.addWidget(self.progress_bar)
        slider_layout.addWidget(self.score_label)

        cell_image = QVBoxLayout(self.rightContainer)
        self.cell_image = ClickableLabel(self.rightContainer)
        self.cell_image.setMinimumSize(384, 384)
        self.cell_image.setAlignment(Qt.AlignCenter)
        cell_image.addWidget(self.cell_image)

        rightVerticalLayout.addLayout(self._build_legend())
        rightVerticalLayout.addLayout(cell_image)
        rightVerticalLayout.addLayout(self.synthetic_image)
        rightVerticalLayout.addLayout(self.real_image)
        rightVerticalLayout.addLayout(startStopLayout)
        rightVerticalLayout.addLayout(showResults)
        rightVerticalLayout.addLayout(helpLayout)
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
            user=[0]*len(self.curr_coords),
            user_touched=[False]*len(self.curr_coords)))

        self.curr_masks = resize_with_scipy(self.curr_masks, new_W, new_H)

    LEGEND_ENTRIES = [
        ((64, 64, 64), "NORMAL"),
        ((255, 255, 0), "LOSS"),
        ((0, 255, 255), "GAIN"),
        ((255, 0, 255), "AMP"),
    ]

    def _build_legend(self):
        layout = QHBoxLayout()
        layout.setSpacing(18)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addStretch()

        for (r, g, b), name in self.LEGEND_ENTRIES:
            entry = QHBoxLayout()
            entry.setSpacing(8)

            swatch = QLabel()
            swatch.setFixedSize(28, 28)
            swatch.setStyleSheet(
                f"border: 3px solid rgb({r},{g},{b}); background-color: black;"
            )

            text = QLabel(name)
            text.setStyleSheet("font-size: 13pt; font-weight: bold; color: white;")

            entry.addWidget(swatch)
            entry.addWidget(text)
            layout.addLayout(entry)
            layout.addStretch()

        return layout

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
            user=[0]*len(self.curr_coords),
            user_touched=[False]*len(self.curr_coords)))
        
        self.curr_masks = resize_with_scipy(self.curr_masks, new_W, new_H)

    def connect_signals(self):

        self.synthetic_image_button.clicked.connect(self.create_image)
        self.Image.imageClicked.connect(self.image_clicked)
        self.start.clicked.connect(self.start_timer)
        self.stop.clicked.connect(self.stop_timer)
        self.real_image_button.clicked.connect(self.load_image)
        self.gt_results.clicked.connect(self.gt_clicked)
        self.ai_results.clicked.connect(self.ai_clicked)
        self.user_results.clicked.connect(self.user_clicked)
        self.help_results.clicked.connect(self.help_clicked)

    def _count_on(self):
        return int(self.GT_CLICKED) + int(self.AI_CLICKED) + int(self.USER_CLICKED)

    def _try_toggle(self, attr_name, widget):
        cur = getattr(self, attr_name)
        # allow turning off always; block turning on if already 2 on
        if not cur and self._count_on() >= MAX_ON:
            return  # ignore this click

        if not cur and self.HELP_CLICKED:
            self.HELP_CLICKED = False
            self.help_results.setStyleSheet(HELP_STYLESHEET_OFF)

        new = not cur
        setattr(self, attr_name, new)
        widget.setStyleSheet(STYLED[0] if new else STYLED[1])
        self.switch_show()

    def gt_clicked(self):
        self._try_toggle("GT_CLICKED", self.gt_results)

    def ai_clicked(self):

        if self._inference_running:
            return

        if not self.MODEL_RAN:
            if self.curr_patches is None or len(self.curr_patches) == 0:
                return
            self._ai_toggle_pending = True
            self._help_toggle_pending = False
            self._start_inference()
            return

        self._try_toggle("AI_CLICKED", self.ai_results)

    def user_clicked(self):
        self._try_toggle("USER_CLICKED", self.user_results)

    def help_clicked(self):

        if self._inference_running:
            return

        if not self.MODEL_RAN:
            if self.curr_patches is None or len(self.curr_patches) == 0:
                return
            self._help_toggle_pending = True
            self._ai_toggle_pending = False
            self._start_inference()
            return

        self._toggle_help()

    def _toggle_help(self):
        if not self.HELP_CLICKED:
            for attr, widget in (("GT_CLICKED", self.gt_results),
                                 ("AI_CLICKED", self.ai_results),
                                 ("USER_CLICKED", self.user_results)):
                if getattr(self, attr):
                    setattr(self, attr, False)
                    widget.setStyleSheet(STYLED[1])
            self.HELP_CLICKED = True
            self.help_results.setStyleSheet(HELP_STYLESHEET_ON)
        else:
            self.HELP_CLICKED = False
            self.help_results.setStyleSheet(HELP_STYLESHEET_OFF)

        self.switch_show()

    def switch_show(self):

        if self.HELP_CLICKED:
            self._score_timer.stop()
            self._score_target = None
            self.score_label.setText("")
            self.plot_help()
            return

        n_toggled = (self.GT_CLICKED + self.AI_CLICKED + self.USER_CLICKED)

        if n_toggled != 2:
            self._score_timer.stop()
            self._score_target = None
            self.score_label.setText("")

        if n_toggled == 0:
            self.plot_only_image()

        elif n_toggled == 1:
            self.plot_single_annotations()

        elif n_toggled == 2:
            self.plot_comparison()
        else:
            print('SHOULD NEVER HAPPEN')

    def plot_help(self):
        self.drawRectanglesAndText(mode='help')

    def _update_help_score(self):

        df = self.user_clicked_dict
        if df is None or 'uncertainty' not in df.columns or 'ai' not in df.columns:
            self.score_label.setText("")
            return

        touched = df['user_touched'] if 'user_touched' in df.columns else pd.Series(False, index=df.index)
        uncertain_mask = df['uncertainty'] > HELP_UNCERTAINTY_THRESHOLD
        touched_unc = int((touched & uncertain_mask).sum())
        n_unc = int(uncertain_mask.sum())

        if self.GT_AVAILABLE and 'ground_truth' in df.columns:
            ai_correct = (df['ai'] == df['ground_truth']).sum()
            # Hybrid: override AI with user's label on user-touched uncertain cells
            hybrid_pred = df['ai'].where(~(touched & uncertain_mask), df['user'])
            hybrid_correct = (hybrid_pred == df['ground_truth']).sum()
            n = len(df)

            ai_pct = 100 * ai_correct / n if n else 0
            hy_pct = 100 * hybrid_correct / n if n else 0
            delta = hy_pct - ai_pct
            sign = "+" if delta >= 0 else ""
            color = "#7CFC00" if delta >= 0 else "#FF5050"
            self.score_label.setStyleSheet(
                f"font-size: 15pt; font-weight: bold; color: {color};"
            )
            self.score_label.setText(
                f"AI alone: {ai_pct:.1f}%   →   With your help: {hy_pct:.1f}%  ({sign}{delta:.1f}%)\n"
                f"You have annotated {touched_unc}/{n_unc} uncertain cells"
            )
        else:
            if touched_unc == 0:
                self.score_label.setStyleSheet("font-size: 15pt; font-weight: bold; color: #BBB;")
                self.score_label.setText(f"{n_unc} uncertain cells — click them to give your opinion")
                return
            agree = int(((df['user'] == df['ai']) & touched & uncertain_mask).sum())
            self.score_label.setStyleSheet("font-size: 15pt; font-weight: bold; color: #BBB;")
            self.score_label.setText(
                f"Annotated {touched_unc}/{n_unc}. "
                f"You agree with AI on {agree}/{touched_unc} "
                f"({100*agree/touched_unc:.0f}%)"
            )

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
            self._update_score('ground_truth', 'user')

        if self.GT_CLICKED and self.AI_CLICKED:
            self.drawRectanglesAndText(mode='comparison', comp1='ground_truth', comp2='ai')
            self._update_score('ground_truth', 'ai')

        if self.AI_CLICKED and self.USER_CLICKED:
            self.drawRectanglesAndText(mode='comparison', comp1='ai', comp2='user')
            self._update_score('ai', 'user')

    SCORE_LABELS = {
        ('ground_truth', 'user'): "You",
        ('ground_truth', 'ai'): "AI",
        ('ai', 'user'): "You vs AI",
    }

    def _update_score(self, comp1, comp2):

        if self.user_clicked_dict is None:
            self.score_label.setText("")
            return

        if 'ground_truth' in (comp1, comp2) and not self.GT_AVAILABLE:
            self.score_label.setText("GT not available for real images")
            return

        if comp1 not in self.user_clicked_dict.columns or comp2 not in self.user_clicked_dict.columns:
            self.score_label.setText("")
            return

        a = self.user_clicked_dict[comp1]
        b = self.user_clicked_dict[comp2]
        n = len(a)
        matches = int((a == b).sum())
        tag = self.SCORE_LABELS.get((comp1, comp2), f"{comp1} vs {comp2}")
        self._start_score_countup(tag, matches, n)

    SCORE_DEFAULT_STYLE = "font-size: 18pt; font-weight: bold; color: #7CFC00;"

    def _start_score_countup(self, tag, matches, total):
        self._score_timer.stop()
        self.score_label.setStyleSheet(self.SCORE_DEFAULT_STYLE)
        self._score_target = {
            'tag': tag,
            'matches': matches,
            'total': total,
            'step': 0,
        }
        self._score_timer.start()
        self._score_tick()

    def _score_tick(self):
        if self._score_target is None:
            self._score_timer.stop()
            return

        state = self._score_target
        state['step'] += 1
        t = min(1.0, state['step'] / self._score_steps_total)
        eased = 1 - (1 - t) ** 3  # ease-out cubic
        shown_matches = int(round(eased * state['matches']))
        pct = (100 * shown_matches / state['total']) if state['total'] else 0.0
        self.score_label.setText(f"{state['tag']}: {shown_matches}/{state['total']} ({pct:.1f}%)")

        if t >= 1.0:
            self._score_timer.stop()
            self._score_target = None


    def reset_clicked(self):

        self.GT_CLICKED = self.USER_CLICKED = self.AI_CLICKED = False
        for widget in [self.gt_results, self.user_results, self.ai_results]:
            widget.setStyleSheet(STYLED[1])
        self.HELP_CLICKED = False
        self.help_results.setStyleSheet(HELP_STYLESHEET_OFF)
        self.timer_label.setText("")
        self.score_label.setText("")
        self.score_label.setStyleSheet(self.SCORE_DEFAULT_STYLE)
        self.ANNOTATED = False
        self.timer_text = ""

        self._ai_reveal_timer.stop()
        self._ai_animation_active = False
        self._ai_revealed_count = 0

        self._score_timer.stop()
        self._score_target = None


    def _start_inference(self):

        self._inference_running = True
        self._ai_toggle_pending = True
        self._inference_patches_id = id(self.user_clicked_dict)
        self._set_inference_ui_enabled(False)
        self._set_timer_suffix("Running AI...")

        total = len(self.curr_patches)
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

        worker = InferenceWorker(self.model, np.array(self.curr_patches), parent=self)
        worker.finished_ok.connect(self._on_inference_done)
        worker.failed.connect(self._on_inference_failed)
        worker.progress.connect(self._on_inference_progress)
        worker.finished.connect(worker.deleteLater)
        worker.finished.connect(self._on_worker_finished)
        self._inference_worker = worker
        worker.start()

    def _on_inference_progress(self, done, total):
        if total != self.progress_bar.maximum():
            self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(done)

    def _on_worker_finished(self):
        self._inference_running = False
        self._inference_worker = None

    def _set_inference_ui_enabled(self, enabled):
        self.ai_results.setEnabled(enabled)
        self.synthetic_image_button.setEnabled(enabled)
        self.real_image_button.setEnabled(enabled)

    def _on_inference_done(self, results, elapsed):

        self.progress_bar.setVisible(False)

        if id(self.user_clicked_dict) != self._inference_patches_id:
            self._set_timer_suffix("AI discarded (image changed)", commit=True)
            self._set_inference_ui_enabled(True)
            self._ai_toggle_pending = False
            self._help_toggle_pending = False
            return

        self.model_timer = elapsed
        self.user_clicked_dict['ai'] = [CLASS_MAP[c] for c in results['classification']]
        self.user_clicked_dict['regression'] = results['regression']
        self.user_clicked_dict['uncertainty'] = results['uncertainty']

        self._set_timer_suffix(f"Model took {elapsed:.2f}s", commit=True)
        self.MODEL_RAN = True
        self._set_inference_ui_enabled(True)

        self._setup_ai_reveal()

        if self._ai_toggle_pending:
            self._ai_toggle_pending = False
            self._try_toggle("AI_CLICKED", self.ai_results)
        elif self._help_toggle_pending:
            self._help_toggle_pending = False
            self._toggle_help()

        if self._ai_animation_active and self.AI_CLICKED:
            self._ai_reveal_timer.start()

    def _setup_ai_reveal(self):
        coords = np.stack([np.asarray(c) for c in self.user_clicked_dict['coords'].values])
        self._ai_revealed_order = np.argsort(coords[:, 0])
        self._ai_revealed_count = 0
        self._ai_revealed_total = len(self._ai_revealed_order)
        self._ai_animation_active = self._ai_revealed_total > 0

    def _ai_reveal_tick(self):
        if not self._ai_animation_active:
            self._ai_reveal_timer.stop()
            return

        step = max(1, self._ai_revealed_total // 40)
        self._ai_revealed_count = min(self._ai_revealed_total, self._ai_revealed_count + step)

        if self._ai_revealed_count >= self._ai_revealed_total:
            self._ai_animation_active = False
            self._ai_reveal_timer.stop()

        if self.AI_CLICKED:
            self.switch_show()

    def _on_inference_failed(self, err):

        self.progress_bar.setVisible(False)
        self._set_timer_suffix(f"AI failed: {err}", commit=True)
        self._set_inference_ui_enabled(True)
        self._ai_toggle_pending = False
        self._help_toggle_pending = False

    def _set_timer_suffix(self, suffix, commit=False):
        """Preview (commit=False) or persist (commit=True) a suffix to the timer label."""
        base = self.timer_text
        text = (base + " / " + suffix) if base else suffix
        if commit:
            self.timer_text = text
        self.timer_label.setText(text)

    def closeEvent(self, event):
        w = self._inference_worker
        if self._inference_running and w is not None:
            w.quit()
            w.wait(2000)
        super().closeEvent(event)


    def start_timer(self):
    
        if self.ANNOTATION_MODE or self.ANNOTATED:
            return 
        self.ANNOTATION_MODE = True
        
        self.timer = time()
        
    def stop_timer(self):
        """ Stops the timer and updates the summary row with user and model statistics """
        
        if not self.ANNOTATION_MODE:
            return
        self.ANNOTATION_MODE = False
        self.ANNOTATED = True

        self.timer = time() - self.timer

        user_text = f"User took {self.timer:.2f}s"

        if self.timer_text:
            self.timer_text += " / " + user_text
        else: 
            self.timer_text += user_text

        self.timer_label.setText(self.timer_text)
  
    def image_clicked(self, pos):

        if self.curr_masks is None or self.user_clicked_dict is None:
            return

        if self.curr_masks[pos.y(), pos.x()] == 0:
            return

        min_idx = int(np.argmin(cdist(self.curr_coords, np.atleast_2d(np.array([pos.x(), pos.y()])))))
        label = self.user_clicked_dict.index[min_idx]

        in_help = self.HELP_CLICKED and 'uncertainty' in self.user_clicked_dict.columns
        cell_is_uncertain = (
            in_help
            and self.user_clicked_dict.at[label, 'uncertainty'] > HELP_UNCERTAINTY_THRESHOLD
        )

        self.only_show = True
        if (self.AI_CLICKED + self.USER_CLICKED + self.GT_CLICKED) == 0 and self.ANNOTATION_MODE:
            self.only_show = False
        if cell_is_uncertain:
            self.only_show = False

        if not self.only_show:
            cur = self.user_clicked_dict.at[label, 'user']
            self.user_clicked_dict.at[label, 'user'] = 0 if cur == 3 else cur + 1
            self.user_clicked_dict.at[label, 'user_touched'] = True

        self.tmp = qimage2ndarray.array2qimage(resize_with_scipy(self.user_clicked_dict.at[label, 'patches']*255, 384, 384))
        self.cell_image.setPixmap(QtGui.QPixmap.fromImage(self.tmp))

        if self.HELP_CLICKED:
            self.switch_show()
        elif not self.ANNOTATION_MODE:
            self.switch_show()
        else:
            self.drawRectanglesAndText(mode='user')
    
            

    def drawRectanglesAndText(self, mode, comp1=None, comp2=None):


        _user_clicked_dict = self.user_clicked_dict.copy()  # rename to df if it’s really a DataFrame

        if mode == 'ai' and self._ai_animation_active and self._ai_revealed_order is not None:
            visible = set(self._ai_revealed_order[:self._ai_revealed_count].tolist())
            _user_clicked_dict = _user_clicked_dict[_user_clicked_dict.index.isin(visible)]

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

        if mode == 'help':

            if 'uncertainty' not in _user_clicked_dict.columns:
                painter.end()
                self.Image.setPixmap(pixmap)
                return

            _user_clicked_dict = _user_clicked_dict[
                _user_clicked_dict['uncertainty'] > HELP_UNCERTAINTY_THRESHOLD
            ]

            help_pen = QPen(QColor(255, 60, 60), 4)
            help_pen.setStyle(Qt.DashLine)

            for _, row in _user_clicked_dict.iterrows():
                rectangle = QRect(row.coords[0]-34, row.coords[1]-34, 68, 68)
                painter.setPen(help_pen)
                painter.drawRect(rectangle)

                ai_text = f"AI: ?\n{1 - row['uncertainty']:.0%}"
                painter.setPen(QColor(255, 200, 200))
                fm = painter.fontMetrics()
                text_rect = fm.boundingRect(QRect(0, 0, 0, 0), Qt.AlignCenter | Qt.TextWordWrap, ai_text)
                text_position = QPoint(row.coords[0], row.coords[1]-46)
                text_rect.moveCenter(text_position)
                painter.drawText(text_rect, Qt.AlignCenter | Qt.TextWordWrap, ai_text)

                if bool(row.get('user_touched', False)):
                    user_class_idx = int(row['user'])
                    user_color = colors_dict[user_class_idx]['color']
                    user_name = colors_dict[user_class_idx]['class']

                    inner_pen = QPen(user_color, 3)
                    painter.setPen(inner_pen)
                    painter.drawRect(QRect(row.coords[0]-27, row.coords[1]-27, 54, 54))

                    user_text = f"You: {user_name}"
                    painter.setPen(user_color)
                    user_rect = fm.boundingRect(QRect(0, 0, 0, 0), Qt.AlignCenter | Qt.TextWordWrap, user_text)
                    user_pos = QPoint(row.coords[0], row.coords[1]+46)
                    user_rect.moveCenter(user_pos)
                    painter.drawText(user_rect, Qt.AlignCenter | Qt.TextWordWrap, user_text)

            painter.end()
            self.Image.setPixmap(pixmap)
            self._update_help_score()
            return

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
                    color = colors_dict[row.ground_truth]['color']
                    class_text = colors_dict[row.ground_truth]['class']

                rectangle = QRect(row.coords[0]-32, row.coords[1]-32, 64, 64)
                if mode == 'ai':
                    confidence = float(1 - row['uncertainty'])
                    pen_width = max(1, int(round(1 + 4 * confidence)))
                else:
                    pen_width = 2
                painter.setPen(QPen(color, pen_width))
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
