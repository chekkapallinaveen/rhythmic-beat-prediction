import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load the audio file
file_path = input("Enter the path of the audio file: ")
y, sr = librosa.load(file_path)

hop_size = 0.1
hop_length = int(hop_size * sr)

# Create a spectrogram
D = librosa.amplitude_to_db(librosa.stft(y, hop_length=hop_length), ref=np.max)

# Plot the spectrogram
plt.figure(figsize=(10, 6))
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram of the Music File')
plt.show()