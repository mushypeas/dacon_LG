import cv2
import json
from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

files = sorted(glob('data/train/13674'))
ois = pd.read_csv('data/train_4.csv')['image']

total_images = len(files)
image_sizes = {}
count = 0
for file in ois:
    count += 1
    if count < 440:
        continue
    crop_image = cv2.imread(f'data/train/{file}/{file}.jpg')
    # visualize bbox
    cv2.imshow(f"{file}", crop_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"Images[{count}/{total_images}]")
