import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def plot_amplitude_and_frequency(audio_file):
  # Load audio file
  y, sr = librosa.load(audio_file)

  # Time-domain representation (Amplitude)
  plt.figure(figsize=(12, 8))
  plt.subplot(2, 1, 1)
  librosa.display.waveshow(y, sr=sr)
  plt.title('Amplitude Envelope')
  plt.xlabel('Time (s)')
  plt.ylabel('Amplitude')

  # Frequency-domain representation (Spectrogram)
  plt.subplot(2, 1, 2)
  D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
  librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
  plt.colorbar(format='%+2.0f dB')
  plt.title('Spectrogram')
  plt.xlabel('Time (s)')
  plt.ylabel('Frequency (Hz)')

  plt.tight_layout()
  plt.show()

# Example usage
audio_file = "CMR_subset_1.0/audio/01_10003_1-04_Shri_Visvanatham.wav"
plot_amplitude_and_frequency(audio_file)