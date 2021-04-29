import argparse
import os
import pandas as pd
import numpy as np
import librosa
import audio_parameters as ap
import soundfile as sf
from scipy import signal

def vocode(file_list, mels_dir):
    df = pd.read_csv(file_list, sep='|', header=None)
    for _, ex_name in df[0].items():
        mag = np.load(os.path.join(mels_dir, "{}.mag.npy".format(ex_name)))
        # transpose
        mag = mag.T

        # de-noramlize
        mag = (np.clip(mag, 0, 1) * ap.max_db) - ap.max_db + ap.ref_db

        # to amplitude
        mag = np.power(10.0, mag * 0.05)

        # wav reconstruction
        wav = librosa.griffinlim(mag**ap.power, hop_length=ap.hop_length, win_length=ap.win_length)

        # de-preemphasis
        wav = signal.lfilter([1], [1, -ap.preemphasis], wav)

        # trim
        wav, _ = librosa.effects.trim(wav)

        sf.write(os.path.join(mels_dir, "{}_dup.wav".format(ex_name)), wav, ap.sr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_list', help='csv file with a list of all the LJ examples to vocode')
    parser.add_argument('--mels_dir', help='dir with all the mels to predict')
    args = parser.parse_args()
    
    vocode(args.file_list, args.mels_dir)

