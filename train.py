import os
os.environ["WANDB_DISABLED"] = "True"

import pandas as pd
import numpy as np
import cv2
import shutil
import yaml
import warnings
warnings.filterwarnings("ignore")

from ultralytics import YOLO
from glob import glob
from tqdm import tqdm
from IPython.display import clear_output
from sklearn.model_selection import train_test_split

BATCH_SIZE = 2
MODEL = "dacon"

model = YOLO("yolov8x")
results = model.train(
    data="data/yolo/custom.yaml",
    imgsz=(1024, 1024),
    epochs=200,
    batch=BATCH_SIZE,
    project=f"{MODEL}",
    )