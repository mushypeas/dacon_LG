import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
import json 
import torch
from torch import nn
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

device = torch.device("cuda:0")

label_json = json.load(open("labels.json", "r"))
crops, diseases, risks = {}, {}, {}
_crops, _diseases, _risks = label_json["crop"], label_json["disease"], label_json["risk"]
for _crop in _crops:
    crops[int(_crop)] = _crops[_crop]
for _disease in _diseases:
    diseases[int(_disease)] = _diseases[_disease]
for _risk in _risks:
    risks[int(_risk)] = _risks[_risk]
 
def accuracy_function(real, pred):    
    real = real.cpu()
    pred = torch.argmax(pred, dim=1).cpu()
    score = f1_score(real, pred, average='macro')
    return score

class CropClassifier(nn.Module):
    def __init__(self, num_classes, rate):
        super(CropClassifier, self).__init__()
        self.model = models.efficientnet_b1(pretrained=True)
        self.final_layer = nn.Linear(1000, num_classes)
        self.dropout = nn.Dropout(rate)

    def forward(self, image):
        hidden = self.model(image)
        output = self.dropout((self.final_layer(hidden)))
        return output

class CropModel():
    def __init__(self, dataset, dataloader, num_classes, dropout_rate, learning_rate):
        self.train_dataset, self.pred_dataset, self.val_dataset, self.test_dataset = dataset
        self.train_dataloader, self.pred_dataloader, self.val_dataloader, self.test_dataloader = dataloader
        self.model = CropClassifier(num_classes=num_classes, rate=dropout_rate).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.loss_plot = []
        self.val_loss_plot = []
        self.metric_plot = []
        self.val_metric_plot = []
        self.save_path = 'crop_model.pt'

    def train_step(self, batch_item, training):
        img = batch_item['img'].to(device)
        label = batch_item['label'].to(device)
        if training is True:
            self.model.train()
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                output = self.model(img)
                loss = self.criterion(output, label)
            loss.backward()
            self.optimizer.step()
            score = accuracy_function(label, output)
            return loss, score
        else:
            self.model.eval()
            with torch.no_grad():
                output = self.model(img)
                loss = self.criterion(output, label)
            score = accuracy_function(label, output)
            return loss, score

    def train(self, epochs):
        for epoch in range(epochs):
            total_loss, total_val_loss = 0, 0
            total_acc, total_val_acc = 0, 0
            
            tqdm_dataset = tqdm(enumerate(self.train_dataloader))
            for batch, batch_item in tqdm_dataset:
                batch_loss, batch_acc = self.train_step(batch_item, training=True)
                total_loss += batch_loss
                total_acc += batch_acc
                
                tqdm_dataset.set_postfix({
                    'Epoch': epoch + 1,
                    'Loss': '{:06f}'.format(batch_loss.item()),
                    'Mean Loss' : '{:06f}'.format(total_loss/(batch+1)),
                    'Mean F-1' : '{:06f}'.format(total_acc/(batch+1))
                })
            self.loss_plot.append(total_loss/(batch+1))
            self.metric_plot.append(total_acc/(batch+1))
            
            tqdm_dataset = tqdm(enumerate(self.val_dataloader))
            for batch, batch_item in tqdm_dataset:
                batch_loss, batch_acc = self.train_step(batch_item, training=False)
                total_val_loss += batch_loss
                total_val_acc += batch_acc
                
                tqdm_dataset.set_postfix({
                    'Epoch': epoch + 1,
                    'Val Loss': '{:06f}'.format(batch_loss.item()),
                    'Mean Val Loss' : '{:06f}'.format(total_val_loss/(batch+1)),
                    'Mean Val F-1' : '{:06f}'.format(total_val_acc/(batch+1))
                })
            self.val_loss_plot.append(total_val_loss/(batch+1))
            self.val_metric_plot.append(total_val_acc/(batch+1))
            
            if np.max(self.val_metric_plot) == self.val_metric_plot[-1]:
                torch.save(self.model.state_dict(), self.save_path)

    # Loads model from save_path
    def load_model(self):
        self.model = CropClassifier(num_classes=self.num_classes, rate=self.dropout_rate).to(device)
        self.model.load_state_dict(torch.load(self.save_path, map_location=device))
        self.model.to(device)

    def predict(self, output_dict, final=False):
        self.model.eval()
        if final:
            tqdm_dataset = tqdm(enumerate(self.test_dataloader))
        else:
            tqdm_dataset = tqdm(enumerate(self.pred_dataloader))
        for batch, batch_item in tqdm_dataset:
            img = batch_item['img'].to(device)
            nums = batch_item['num'].to(device)
            with torch.no_grad():
                output = self.model(img)
            output = torch.tensor(torch.argmax(output, dim=1), dtype=torch.int32).cpu().numpy()
            for i in range(len(nums)):
                num = int(nums[i])
                label = output_dict[num].split('_')
                output_dict[num] = f"{output[i]+1}_{label[1]}_{label[2]}"
        return output_dict

class CNN_Encoder(nn.Module):
    def __init__(self):
        super(CNN_Encoder, self).__init__()
        self.model = models.efficientnet_b5(pretrained=True)
    
    def forward(self, inputs):
        output = self.model(inputs)
        return output

class RNN_Decoder(nn.Module):
    def __init__(self, max_len, embedding_dim, lstm_dim, num_features, num_classes, rate):
        super(RNN_Decoder, self).__init__()
        self.lstm = nn.LSTM(max_len, embedding_dim)
        self.rnn_fc = nn.Linear(num_features*embedding_dim, lstm_dim)
        self.final_layer = nn.Linear(1000 + lstm_dim, num_classes) # cnn out_dim + lstm out_dim
        self.dropout = nn.Dropout(rate)

    def forward(self, enc_out, dec_inp):
        hidden, _ = self.lstm(dec_inp)
        hidden = hidden.view(hidden.size(0), -1)
        hidden = self.rnn_fc(hidden)
        concat = torch.cat([enc_out, hidden], dim=1) # enc_out + hidden 
        output = self.dropout((self.final_layer(concat)))
        return output

class DiseaseClassifier(nn.Module):
    def __init__(self, max_len, embedding_dim, lstm_dim, num_features, num_classes, rate):
        super(DiseaseClassifier, self).__init__()
        self.cnn = CNN_Encoder()
        self.rnn = RNN_Decoder(max_len, embedding_dim, lstm_dim, num_features, num_classes, rate)
        
    def forward(self, img, seq):
        cnn_output = self.cnn(img)
        output = self.rnn(cnn_output, seq)
        
        return output

# Each crop has its own model for disease detection
# Batch size of DiseaseModel is fixed to 1.
class DiseaseModel():
    def __init__(self, dataset, dataloader, max_len, embedding_dim, lstm_dim, num_features, dropout_rate, learning_rate, crop=None):
        self.train_dataset, self.val_dataset, self.test_dataset = dataset
        self.train_dataloader, self.val_dataloader, self.test_dataloader = dataloader
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.lstm_dim = lstm_dim
        self.num_features = num_features
        self.dropout_rate = dropout_rate
        self.crop = crop
        self.model = DiseaseClassifier(max_len, embedding_dim, lstm_dim, num_features, len(diseases[crop])*len(risks), dropout_rate).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.save_path = f'disease_model{crop}.pt'
        self.loss_plot, self.val_loss_plot = [], []
        self.metric_plot, self.val_metric_plot = [], []

    def train_step(self, batch_item, training):
        img = batch_item['img'].to(device)
        label = batch_item['label'].to(device)
        csv_feature = batch_item['csv_feature'].to(device)
        model = self.model
        optimizer = self.optimizer
        if training is True:
            model.train()
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                output = model(img, csv_feature)
                loss = self.criterion(output, label)
            loss.backward()
            optimizer.step()
            score = accuracy_function(label, output)
            return loss, score
        else:
            model.eval()
            with torch.no_grad():
                output = model(img, csv_feature)
                loss = self.criterion(output, label)
            score = accuracy_function(label, output)
            return loss, score
            
    def train(self, epochs):
        for epoch in range(epochs):
            total_loss, total_val_loss = {}, {}
            total_acc, total_val_acc = {}, {}
            total_loss, total_val_loss, total_acc, total_val_acc = 0, 0, 0, 0

            tqdm_dataset = tqdm(enumerate(self.train_dataloader))
            for batch, batch_item in tqdm_dataset:
                batch_loss, batch_acc = self.train_step(batch_item, training=True)
                total_loss += batch_loss
                total_acc += batch_acc
                tqdm_dataset.set_postfix({
                    'Crop': self.crop,
                    'Epoch': epoch + 1,
                    'Loss': '{:06f}'.format(batch_loss.item()),
                    'Mean Loss' : '{:06f}'.format(total_loss/(batch+1)),
                    'Mean F-1' : '{:06f}'.format(total_acc/(batch+1))
                })
            self.loss_plot.append(total_loss/(batch+1))
            self.metric_plot.append(total_acc/(batch+1))
            
            tqdm_dataset = tqdm(enumerate(self.val_dataloader))
            for batch, batch_item in tqdm_dataset:
                batch_loss, batch_acc = self.train_step(batch_item, training=False)
                total_val_loss += batch_loss
                total_val_acc += batch_acc
                tqdm_dataset.set_postfix({
                    'Crop': self.crop,
                    'Epoch': epoch + 1,
                    'Val Loss': '{:06f}'.format(batch_loss.item()),
                    'Mean Val Loss' : '{:06f}'.format(total_val_loss/(batch+1)),
                    'Mean Val F-1' : '{:06f}'.format(total_val_acc/(batch+1))
                })
            self.val_loss_plot.append(total_val_loss/(batch+1))
            self.val_metric_plot.append(total_val_acc/(batch+1))
            
            if np.max(self.val_metric_plot) == self.val_metric_plot[-1]:
                torch.save(self.model.state_dict(), self.save_path)

    # Loads model from save_path
    def load_model(self):
        self.model = DiseaseClassifier(
            max_len=self.max_len,
            embedding_dim=self.embedding_dim,
            lstm_dim=self.lstm_dim,
            num_features=self.num_features,
            num_classes=len(diseases[self.crop])*len(risks),
            rate=self.dropout_rate
        ).to(device)
        self.model.load_state_dict(torch.load(self.save_path, map_location=device))
        self.model.to(device)

    def predict(self, output_dict, decoder):
        self.model.eval()
        tqdm_dataset = tqdm(enumerate(self.test_dataloader))
        for batch, batch_item in tqdm_dataset:
            img = batch_item['img'].to(device)
            seq = batch_item['csv_feature'].to(device)
            nums = batch_item['num'].to(device)
            with torch.no_grad():
                output = self.model(img, seq)
            output = torch.tensor(torch.argmax(output, dim=1), dtype=torch.int32).cpu().numpy()
            for i in range(len(nums)):
                num = int(nums[i])
                label = output_dict[num].split('_')
                output_dict[num] = f"{label[0]}_{decoder[self.crop][output[i]]}"
        return output_dict
