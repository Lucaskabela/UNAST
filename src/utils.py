'''
Contains any and all code we didnt want to put somewhere else
'''
import argparse
import torch
import numpy as np 
import random

def parse_args():
    parser = argparse.ArgumentParser(from_file_prefix_chars="@")

    parser.add_argument('--seed', type=int, default=0, help='RNG seed (default = 0)')
    parser.add_argument('--epochs', type=int, default=100, help='num epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    
    return parser.parse_args('@hyperparams.txt')


def set_seed(seed):
    '''
    Sets torch, numpy, and random library with seed for reproducibility
    See: https://pytorch.org/docs/stable/notes/randomness.html for more details
    on setting determinism

    Args:
        - seed: An integer seed for consistency in different runs
    ''' 
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
import numpy as np
import librosa
import audio_parameters as ap
import json
import sys

def parse_with_config(parser):
    """
        Method that takes a parser with a --config flag that takes the json file
        and returns a namespace with attributes from the json file.
        eg.
        Call `file.py --config config.json`
        where config.json looks like
        `{
            parameter1: 10
        }`
        then this returns `args`
        where `args.parameter1` = 10

        **** Doesn't work with nested objects, everything in the json file
        should only be one level deep

        Args:
            An ArgumentParser with a --config flag
        Returns:
            A namespace with the keys in the json file as attributes

    """
    args = parser.parse_args()
    if args.config is not None:
        config_args = json.load(open(args.config))
        override_keys = {arg[2:].split('=')[0] for arg in sys.argv[1:]
                         if arg.startswith('--')}
        for k, v in config_args.items():
            if k not in override_keys:
                setattr(args, k, v)
    del args.config
    return args

def get_spectrograms(fpath):
    '''Parse the wave file in `fpath` and
    Returns normalized melspectrogram and linear spectrogram.
    Args:
      fpath: A string. The full path of a sound file.
    Returns:
      mel: A 2d array of shape (T, n_mels) and dtype of float32.
      mag: A 2d array of shape (T, 1+n_fft/2) and dtype of float32.
    '''
    # Loading sound file
    y, sr = librosa.load(fpath, sr=ap.sr)

    # Trimming
    y, _ = librosa.effects.trim(y)

    # Preemphasis
    y = np.append(y[0], y[1:] - ap.preemphasis * y[:-1])

    # stft
    linear = librosa.stft(y=y,
                          n_fft=ap.n_fft,
                          hop_length=ap.hop_length,
                          win_length=ap.win_length)

    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2, T)

    # mel spectrogram
    mel_basis = librosa.filters.mel(ap.sr, ap.n_fft, ap.n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t)

    # to decibel
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))

    # normalize
    mel = np.clip((mel - ap.ref_db + ap.max_db) / ap.max_db, 1e-8, 1)
    mag = np.clip((mag - ap.ref_db + ap.max_db) / ap.max_db, 1e-8, 1)

    # Transpose
    mel = mel.T.astype(np.float32)  # (T, n_mels)
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)

    return mel, mag
