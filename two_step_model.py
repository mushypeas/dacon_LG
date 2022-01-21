import os
from re import sub
os.environ["CUDA_LAUNCH_BLOCKING"]="1"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from custom_models import CropModel, DiseaseModel
from custom_dataset import CropDataset, DiseaseDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from glob import glob
import json 
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

label_json = json.load(open("labels.json", "r"))
crops, diseases, risks = {}, {}, {}
_crops, _diseases, _risks = label_json["crop"], label_json["disease"], label_json["risk"]
for _crop in _crops:
    crop = int(_crop)
    crops[crop] = _crops[_crop]
    diseases[crop] = _diseases[_crop]
for _risk in _risks:
    risks[int(_risk)] = _risks[_risk]

disease_label_encoder, disease_label_decoder = {}, {}
for crop in crops:
    disease_label_encoder[crop], disease_label_decoder[crop] = {}, {}
    for idx, key in enumerate((diseases[crop])):
        if idx == 0:
            label = '00_0'
            disease_label_encoder[crop][label] = 0
            disease_label_decoder[crop][0] = label
        else:
            for risk in risks:
                label = f'{key}_{risk}'
                code = (idx-1)*len(risks) + risk
                disease_label_encoder[crop][label] = code
                disease_label_decoder[crop][code] = label

device = torch.device("cuda:0")
num_crops = len(crops)

# Hyperparameters
batch_size = 32
learning_rate = 1e-4
embedding_dim = 512
max_len = 24*6
dropout_rate = 0.1
num_workers = 32
epochs = 20
lstm_dim = 100
vision_pretrain = True


# # Step 1: Classify crop
# print("Step 1: Classify Crop")

# train = sorted(glob('data/train/*'))
# test = sorted(glob('data/test/*'))
# pred = train

# labelsss = pd.read_csv('data/train.csv')['label']
# train, val = train_test_split(train, test_size=0.2, stratify=labelsss)

# crop_train_dataset = CropDataset(train)
# crop_pred_dataset = CropDataset(pred)
# crop_val_dataset = CropDataset(val)
# crop_test_dataset = CropDataset(test, mode='test')
# crop_dataset = crop_train_dataset, crop_pred_dataset, crop_val_dataset, crop_test_dataset

# crop_train_dataloader = DataLoader(crop_train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
# crop_pred_dataloader = DataLoader(crop_pred_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
# crop_val_dataloader = DataLoader(crop_val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
# crop_test_dataloader = DataLoader(crop_test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
# crop_dataloader = crop_train_dataloader, crop_pred_dataloader, crop_val_dataloader, crop_test_dataloader

# crop_model = CropModel(crop_dataset, crop_dataloader, num_crops, dropout_rate, learning_rate)
# # crop_model.train(epochs=epochs)
# crop_model.load_model()

# prediction = pd.read_csv('data/train.csv')
# prediction_dict = {}
# for idx, row in prediction.iterrows():
#     prediction_dict[int(row['image'])] = '0_00_0'

# prediction_dict = crop_model.predict(prediction_dict)

# # Create prediction file
# prediction['image'] = np.array(prediction_dict.keys())
# prediction['label'] = np.array([prediction_dict[key] for key in prediction_dict.keys()])
# prediction.to_csv('prediction.csv', index=False)

# Step 2: Classify disease
print("Step 2: Classify Disease")

submission = pd.read_csv('baseline_submission.csv')
submission_dict = {}
for idx, row in submission.iterrows():
    submission_dict[int(row['image'])] = row['label']

prediction = pd.read_csv(f'prediction.csv')
prediction_dict = {}
for idx, row in prediction.iterrows():
    prediction_dict[int(row['image'])] = row['label']
    
for crop in crops:
    # if crop not in [4]:
    #     continue
    print(f"Crop: {crop}")
    train_csv = pd.read_csv(f'data/train_{crop}.csv')
    train_dict = {}
    for idx, row in train_csv.iterrows():
        train_dict[int(row['image'])] = row['label']
    train, val = [], []
    for image in train_dict:
        train.append(f"data/train/{image}")
    train = sorted(train)

    test_csv = pd.read_csv(f'data/submission_{crop}.csv')
    test_dict = {}
    for idx, row in test_csv.iterrows():
        test_dict[int(row['image'])] = row['label']
    test = []
    for image in test_dict:
        test.append(f"data/test/{image}")

    labelsss = pd.read_csv(f'data/train_{crop}.csv')['label']
    # train, val = train_test_split(train, test_size=0.2, stratify=labelsss)
    val = train

    disease_train_dataset = DiseaseDataset(train, data=prediction_dict, encoder=disease_label_encoder)
    disease_val_dataset = DiseaseDataset(val, data=prediction_dict, encoder=disease_label_encoder)
    disease_test_dataset = DiseaseDataset(test, encoder=disease_label_encoder, mode='test')
    disease_dataset = disease_train_dataset, disease_val_dataset, disease_test_dataset

    disease_train_dataloader = DataLoader(disease_train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    disease_val_dataloader = DataLoader(disease_val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    disease_test_dataloader = DataLoader(disease_test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    disease_dataloader = disease_train_dataloader, disease_val_dataloader, disease_test_dataloader

    num_features = len(disease_train_dataset.csv_feature_dict)
    disease_model = DiseaseModel(
        dataset=disease_dataset,
        dataloader=disease_dataloader,
        max_len=max_len,
        embedding_dim=embedding_dim,
        lstm_dim=lstm_dim,
        num_features=num_features,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        crop=crop
    )
    # disease_model.train(epochs=epochs)
    disease_model.load_model()

    test_dict = disease_model.predict(test_dict, decoder=disease_label_decoder)
    for image in test_dict:
        submission_dict[image] = test_dict[image]

# Create submision file
submission['image'] = np.array(submission_dict.keys())
submission['label'] = np.array([submission_dict[key] for key in submission_dict.keys()])
submission.to_csv('baseline_submission.csv', index=False)
