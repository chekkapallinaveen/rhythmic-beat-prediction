# input the path to the audio file
# output the predicted label

from joblib import load

model = load('/Users/chnaveen/Documents/sem5/music/rhythmic-beat-prediction/scripts/dt_amp_mag_classifier.joblib')
lc = load('/Users/chnaveen/Documents/sem5/music/rhythmic-beat-prediction/scripts/label_encoder.joblib')

in_song = input("Enter the path to the audio file: ")

# convert the audio file to the wav format
from pydub import AudioSegment
from pydub.utils import make_chunks

def convert_to_wav(file_path):
  if file_path.endswith('.wav'):
    return file_path
  else:
    song = AudioSegment.from_file(file_path)
    song = song.set_channels(1)
    song = song.set_frame_rate(22050)
    song = song.set_sample_width(2)
    song = song.set_channels(1)
    song = song.set_frame_rate(22050)
    song = song.set_sample_width(2)
    song = song.set_channels(1)
    song = song.set_frame_rate(22050)
    song = song.set_sample_width(2)
    song.export("temp.wav", format="wav")
    return "temp.wav"

# data processing and feature extraction
import librosa
import numpy as np

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

in_song = convert_to_wav(in_song)

amps, mags, mfccs, chromas = extract_features(in_song)

features = []

mfccs_new = np.array(mfccs).reshape(1200, 13*5)
chromas_new = np.array(chromas).reshape(1200, 12*5)

mfccs_new = np.mean(mfccs_new, axis=1)
chromas_new = np.mean(chromas_new, axis=1)

features = np.hstack((amps, mags, mfccs_new, chromas_new))

features = np.array(features).reshape(1, -1)

def predict_song(features):
  return model.predict(features)

print(lc.inverse_transform(predict_song(features)))