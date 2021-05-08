'''
Contains the code for training the vocoder

This code is based off of the train_postnet.py in Transformer-TTS repo.
'''
from utils import *
from preprocess import get_test_mel_dataset, collate_fn_postnet
from network import Vocoder
from train import get_linear_schedule_with_warmup, get_transformer_paper_schedule
import torch
import torch.nn as nn
import argparse
import datetime
import glob
import os

import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from scipy.io.wavfile import write


def initialize(args):
    set_seed(args.seed)

    # Model
    model = Vocoder(args.num_mels, args.hidden_size, args.n_fft)
    model = model.to(DEVICE)

    optimizer = None
    if args.optim_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim_type == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    s_epoch = 0
    if args.load_path is not None:
        if os.path.isfile(args.load_path):
            s_epoch, _, model, optimizer = load_ckp(args.load_path, model, optimizer)
            print(f"[INFO] Using model trained for {s_epoch} epochs.")
        else:
            print(f"[INFO] Could not find checkpoint '{args.load_path}'.")
            print(f"[INFO] Using randomly initialized model.")

    return model


def make_wavs(args):
    model = initialize(args)
    model = model.eval()
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir, mode=0o755, exist_ok=True)
    pbar = tqdm(glob.glob(os.path.join(args.mel_dir, "*.pt.npy")))
    with torch.no_grad():
        for path in pbar:
            name = path.split("/")[-1][:-len(".pt.npy")]
            if not os.path.isfile(os.path.join(args.out_dir, f"{name}.wav")):
                mel = torch.from_numpy(np.load(path))
                if mel.shape[0] == 1:
                    print(f"{name} mel-spectrogram is too short for synthesis, shape: {mel.shape}")
                else:
                    mel = mel.unsqueeze(0).to(DEVICE)
                    mag_preds = model.forward(mel).squeeze(0).detach().cpu().numpy()
                    print("=========================================================================================")
                    print(mel)
                    print("---------------------------------")
                    print(mag_preds)
                    print("---------------------------------")
                    wav = spectrogram2wav(mag_preds)
                    print(wav)
                    print("=========================================================================================")
                    write(os.path.join(args.out_dir, f"{name}.wav"), args.sample_rate, wav)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='JSON config files')
    args = parse_with_config(parser)

    global DEVICE
    DEVICE = init_device(args)
    print(f"[{datetime.datetime.now()}] Device: {DEVICE}")

    make_wavs(args)

