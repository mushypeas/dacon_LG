import os
import numpy as np
import pandas as pd
import cv2
import json 
import torch
from glob import glob
from tqdm import tqdm
from torch.utils.data import Dataset

class CropDataset(Dataset):
    def __init__(self, files, encoder=None, mode='train'):
        self.mode = mode
        self.files = files
        self.encoder = encoder
        self.max_len = 24 * 6

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, i):
        file = self.files[i]
        file_name = file.split('/')[-1]
        
        # image
        image_path = f'{file}/{file_name}.jpg'
        img = cv2.imread(image_path)
        img = cv2.resize(img, dsize=(512, 256), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32)/255
        img = np.transpose(img, (2,0,1))

        if self.mode == 'train':
            json_path = f'{file}/{file_name}.json'
            with open(json_path, 'r') as f:
                json_file = json.load(f)
            
            label = json_file['annotations']['crop'] - 1
            
            return {
                'img' : torch.tensor(img, dtype=torch.float32),
                'num' : torch.tensor(int(file_name), dtype=torch.int32),
                'label' : torch.tensor(label, dtype=torch.long)
            }
        else:
            return {
                'img' : torch.tensor(img, dtype=torch.float32),
                'num' : torch.tensor(int(file_name), dtype=torch.int32)
            }

class DiseaseDataset(Dataset):
    def __init__(self, files, data=None, encoder=None, mode='train'):
        # 분석에 사용할 feature 선택
        self.csv_feature_dict = None
        if os.path.exists("csv_feature.json"):
            with open("csv_feature.json", "r") as input:
                self.csv_feature_dict = json.load(input)
        else:
            csv_features = ['내부 온도 1 평균','내부 습도 1 평균','내부 이슬점 평균']

            csv_files = sorted(glob('data/train/*/*.csv'))

            temp_csv = pd.read_csv(csv_files[0])[csv_features]
            max_arr, min_arr = temp_csv.max().to_numpy(), temp_csv.min().to_numpy()

            # feature 별 최대값, 최솟값 계산
            for csv in tqdm(csv_files[1:]):
                temp_csv = pd.read_csv(csv)[csv_features]
                temp_csv = temp_csv.replace('-',np.nan).dropna()
                if len(temp_csv) == 0:
                    continue
                temp_csv = temp_csv.astype(float)
                temp_max, temp_min = temp_csv.max().to_numpy(), temp_csv.min().to_numpy()
                max_arr = np.max([max_arr,temp_max], axis=0)
                min_arr = np.min([min_arr,temp_min], axis=0)

            # feature 별 최대값, 최솟값 dictionary 생성
            self.csv_feature_dict = {csv_features[i]:[min_arr[i], max_arr[i]] for i in range(len(csv_features))}

            with open("csv_feature.json", "w") as output:
                json.dump(self.csv_feature_dict, output)

        self.mode = mode
        self.files = files
        self.csv_feature_check = [0]*len(self.files)
        self.csv_features = [None]*len(self.files)
        self.encoder = encoder
        self.data = data
        self.max_len = 24 * 6

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, i):
        file = self.files[i]
        file_name = file.split('/')[-1]
        
        # image
        image_path = f'{file}/{file_name}.jpg'
        img = cv2.imread(image_path)
        img = cv2.resize(img, dsize=(512, 256), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32)/255
        img = np.transpose(img, (2,0,1))

        # csv
        if self.csv_feature_check[i] == 0:
            csv_path = f'{file}/{file_name}.csv'
            df = pd.read_csv(csv_path)[self.csv_feature_dict.keys()]
            df = df.replace('-', 0)
            # MinMax scaling
            for col in df.columns:
                df[col] = df[col].astype(float) - self.csv_feature_dict[col][0]
                df[col] = df[col] / (self.csv_feature_dict[col][1]-self.csv_feature_dict[col][0])
            # zero padding
            pad = np.zeros((self.max_len, len(df.columns)))
            length = min(self.max_len, len(df))
            pad[-length:] = df.to_numpy()[-length:]
            # transpose to sequential data
            csv_feature = pad.T
            self.csv_features[i] = csv_feature
            self.csv_feature_check[i] = 1
        else:
            csv_feature = self.csv_features[i]

        if self.mode == 'train':
            json_path = f'{file}/{file_name}.json'
            with open(json_path, 'r') as f:
                json_file = json.load(f)
            
            crop = json_file['annotations']['crop']
            disease = json_file['annotations']['disease']
            risk = json_file['annotations']['risk']
            label = f'{disease}_{risk}'
            return {
                'img' : torch.tensor(img, dtype=torch.float32),
                'num' : torch.tensor(int(file_name), dtype=torch.int32),
                'csv_feature' : torch.tensor(csv_feature, dtype=torch.float32),
                'label' : torch.tensor(self.encoder[crop][label], dtype=torch.long)
            }
        else:
            return {
                'img' : torch.tensor(img, dtype=torch.float32),
                'num' : torch.tensor(int(file_name), dtype=torch.int32),
                'csv_feature' : torch.tensor(csv_feature, dtype=torch.float32)
            }