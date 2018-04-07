import librosa
import numpy as np

def extractFeature(wav_file):
  y, sr = librosa.load(wav_file, sr=44100)
  audio_len_sec = librosa.core.get_duration(y, sr)
  print("Audio length: {0} seconds".format(audio_len_sec))
  # S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
  mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
  # mfcc_d = librosa.feature.delta(mfcc)
  # mfcc_dd = librosa.feature.delta(mfcc_d)
  nsize = mfcc.shape[1]

  print("Feature length: {0}".format(nsize))
  print("Extraction done!")

def main():
  extractFeature('./data/fart_clean1_mono.wav')

if __name__ == "__main__":
  main()