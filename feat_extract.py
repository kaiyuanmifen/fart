import librosa
import numpy as np

def extractFeature(wav_file):
  y, sr = librosa.load(wav_file, sr=44100)

  # Split and remove silence
  split_begin_end = librosa.effects.split(y)
  y_ = np.array([])
  for start_time, end_time in split_begin_end:
    y_ = np.concatenate([y_, y[start_time:end_time]], axis=0)

  audio_len_sec = librosa.core.get_duration(y_, sr)
  print("Sample length: {0}".format(y_.shape[0]))
  print("Audio length: {0} seconds".format(audio_len_sec))
  mfcc = librosa.feature.mfcc(y=y_, sr=sr, n_mfcc=40)
  nsize = mfcc.shape[1]

  print("Feature length: {0}".format(nsize))
  print("Extraction done!")
  return mfcc

def main():
  # 44100 frames per second
  # 1 feature per 512 frames
  extractFeature('./data/fart_clean1_mono.wav')
  extractFeature('./data/negative_sample_party_mono.wav')
  extractFeature('./data/negative_sample_voice_mono.wav')

if __name__ == "__main__":
  main()