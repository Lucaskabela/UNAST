import argparse
import os
import pandas as pd
import numpy as np
import librosa
import audio_parameters as ap
import soundfile as sf
from scipy import signal

def vocode(list_file, mels_dir, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    df = pd.read_csv(list_file, sep='|', header=None)
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

        sf.write(os.path.join(out_dir, "{}.wav".format(ex_name)), wav, ap.sr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--list_file', help='csv file with a list of all the LJ examples to vocode')
    parser.add_argument('--mels_dir', help='dir with all the mels to predict')
    parser.add_argument('--out_dir', help='dir to output the wavs to')
    args = parser.parse_args()
    
    vocode(args.list_file, args.mels_dir, args.out_dir)

