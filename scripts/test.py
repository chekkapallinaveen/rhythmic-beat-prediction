# input the path to the audio file
# output the predicted label

import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from joblib import load

in_song = input("Enter the path to the audio file: ")

# def convert_to_wav(file_path):
#   if file_path.endswith('.wav'):
#     return file_path
#   else:
#     song = AudioSegment.from_file(file_path)
#     return song

# data processing and feature extraction

import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler

def extract_features(file_path, duration_per_step=0.1, n_fft=1024, n_mfcc=13, hop_length=512):
  amplitudes = []
  magnitude_spectrograms = []
  mfcc_features = []
  chroma_features = []

  # Load audio file
  y, sr = librosa.load(file_path)

  # Calculate the number of steps based on the desired duration
  num_steps = int(len(y) / sr / duration_per_step)
  if num_steps < 1200:
    # padding to make sure all the files have the same length
    y = np.pad(y, (0, 1200 * sr - len(y)), 'constant')
    num_steps = 1200
    
  if num_steps > 1200:
    # truncate the file to the first 1200 steps
    y = y[:1200 * sr]
    num_steps = 1200

  # Iterate over the steps and extract features
  for step in range(num_steps):
    start = int(step * sr * duration_per_step)
    end = int((step + 1) * sr * duration_per_step)

    # Ensure that the end index does not exceed the length of the audio signal
    end = min(end, len(y))

    interval_data = y[start:end]

    # Amplitude (Time-domain representation)
    amplitude = np.mean(interval_data)
    amplitudes.append(amplitude)

    # Frequency (Frequency-domain representation - Magnitude Spectrogram)
    magnitude_spectrogram = np.mean(np.abs(librosa.stft(interval_data, n_fft=n_fft)))
    magnitude_spectrograms.append(magnitude_spectrogram)

    # MFCC features
    mfccf = librosa.feature.mfcc(y=interval_data, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
    mfcc_features.append(mfccf)

    # Chroma features
    chromaf = librosa.feature.chroma_stft(y=interval_data, sr=sr, hop_length=hop_length, n_fft=n_fft)
    chroma_features.append(chromaf)

  return amplitudes, magnitude_spectrograms, mfcc_features, chroma_features

amps, mags, mfccs, chromas = extract_features(in_song, duration_per_step=0.1, n_fft=1024)

mfccs_new = np.array(mfccs).reshape(1200, 13*5)
chromas_new = np.array(chromas).reshape(1200, 12*5)

mfccs_new = np.mean(mfccs_new, axis=1)
chromas_new = np.mean(chromas_new, axis=1)

features = np.hstack((amps, mags, mfccs_new, chromas_new))

features = np.array(features).reshape(1, -1)

import pickle

with open('/Users/chnaveen/Documents/sem5/music/rhythmic-beat-prediction/scripts/dt_amp_mag_classifier.pkl', 'rb') as f:
  loaded_dt_model = pickle.load(f)
  
with open('/Users/chnaveen/Documents/sem5/music/rhythmic-beat-prediction/scripts/label_encoder.pkl', 'rb') as f:
  lc = pickle.load(f)

j_model = load('/Users/chnaveen/Documents/sem5/music/rhythmic-beat-prediction/scripts/dt_amp_mag_classifier.joblib')

print(lc.inverse_transform(loaded_dt_model.predict(features)))
print(lc.inverse_transform(j_model.predict(features)))


class Reshape(nn.Module):
  def __init__(self, shape):
    super(Reshape, self).__init__()
    self.shape = shape

  def forward(self, x):
    return x.view(self.shape)

class CNN(nn.Module):
  def __init__(
    self,
    kernel_size=3,
    stride=1,
    padding=1,
    dropout=0.2,
    learning_rate=0.001,
    batch_size=64,
    num_epochs=5
  ):
    super(CNN, self).__init__()
    
    self.conv1 = nn.Conv2d(
      in_channels=1, 
      out_channels=16, 
      kernel_size=kernel_size, 
      stride=stride,
      padding=padding
    )
    self.relu = nn.ReLU()
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    self.conv2 = nn.Conv2d(
      in_channels=16, 
      out_channels=32,
      kernel_size=kernel_size,
      stride=stride,
      padding=padding
    )
    
    self.dim1 = 10
    self.dim2 = 30
    
    self.adjust_shape = Reshape((-1, 32*self.dim1*self.dim2))
    
    self.dropout = nn.Dropout(p=dropout)
    self.fc = nn.Linear(32*self.dim1*self.dim2, 4)
    self.softmax = nn.Softmax(dim=1)
    
    self.layers = [
      self.conv1,         
      self.pool,          
      self.relu,          
      self.conv2,         
      self.pool,          
      self.relu,          
      self.dropout,       
      self.adjust_shape,  
      self.fc,            
      self.softmax        
    ]
    
    self.criterion = nn.CrossEntropyLoss()
    self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
    self.num_epochs = num_epochs
    self.batch_size = batch_size

  def forward(self, x):
    x = self.conv1(x)
    x = self.pool(x)
    x = self.relu(x)
    
    x = self.conv2(x)
    x = self.pool(x)
    x = self.relu(x)
    
    x = self.dropout(x)
    x = self.adjust_shape(x)
    x = self.fc(x)
    x = self.softmax(x)
    return x
  
  def train_model(self, train_dataset, val_dataset):
    train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)
    
    train_loss, val_loss, accuracy, val_accuracy = 0,0,0,0
    
    for epoch in range(self.num_epochs):
      self.train()
      train_loss = 0.0
      correct = 0
      total = 0
      
      val_loss = 0.0
      val_correct = 0
      val_total = 0
      
      for inputs, labels in train_loader:
        self.optimizer.zero_grad()
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward(retain_graph=True)
        self.optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

      self.eval()
      for inputs, labels in val_loader:
        self.optimizer.zero_grad()
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward(retain_graph=True)
        self.optimizer.step()

        val_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        val_total += labels.size(0)
        val_correct += (predicted == labels).sum().item()

      val_accuracy = val_correct / val_total
      accuracy = correct / total

      print(
        f'Epoch {epoch+1}/{self.num_epochs}, '
        f'Loss(Train): {train_loss:.4f}, '
        f'Accuracy(Train): {accuracy:.2f}, '
        f'Loss(Val): {val_loss:.4f}, '
        f'Accuracy(Val): {val_accuracy:.2f}'
      )
    return train_loss, val_loss, accuracy, val_accuracy
  
  def predict(self, pred_dataset):
    pred_loader = DataLoader(pred_dataset, batch_size=self.batch_size, shuffle=False)
    self.eval()
    predictions = []
    labels_true = []
    for inputs, labels in pred_loader:
      outputs = self(inputs)
      _, predicted = torch.max(outputs.data, 1)
      predictions.extend(predicted.tolist())
      labels_true.extend(labels.numpy())
    return predictions, labels_true
  
  def predict_without_labels(self, pred_dataset):
    pred_loader = DataLoader(pred_dataset, batch_size=self.batch_size, shuffle=False)
    self.eval()
    predictions = []
    for inputs in pred_loader:
      outputs = self(inputs)
      _, predicted = torch.max(outputs.data, 1)
      predictions.extend(predicted.tolist())
    return predictions
  
cnn_model = CNN()
cnn_model.load_state_dict(torch.load('/Users/chnaveen/Documents/sem5/music/rhythmic-beat-prediction/scripts/cnn_model.pt'))

cnn_features = np.array(features).reshape(1, 1, 40, 120)
cnn_model.eval()
preds = cnn_model.predict_without_labels(cnn_features)
print(lc.inverse_transform(preds))