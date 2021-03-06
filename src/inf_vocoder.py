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
import os
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


def initialize(args):
    set_seed(args.seed)

    # Dataset
    dataset = get_test_mel_dataset(os.path.join(args.out_test_dir, 'mels') , args.audio_list_file)

    # Model
    model = Vocoder(args.num_mels, args.hidden_size, args.n_fft)
    model = model.to(DEVICE)

    optimizer = None
    if args.optim_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim_type == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Continue training
    s_epoch = 0
    if args.load_path is not None:
        if os.path.isfile(args.load_path):
            s_epoch, _, model, optimizer = load_ckp(args.load_path, model, optimizer)
        else:
            print(f"[WARN] Could not find checkpoint '{args.load_path}'.")
            print(f"[WARN] Training from initial model...")

    return s_epoch, model, dataset


def make_mags(args):
    assert 300 % args.eval_batch_size == 0, "Eval batch size {} must divide the length of the test set (300) perfectly for the dataloader".format(args.eval_batch_size)
    s_epoch, model, dataset = initialize(args)

    # Make dataloader
    dataloader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn_postnet, drop_last=True, num_workers=args.num_workers)

    pbar = tqdm(dataloader)
    with torch.no_grad():
        for i, data in enumerate(pbar):
            mel, mel_lens, fnames = data
            mel = mel.to(DEVICE)

            mag_preds = model.forward(mel)

            for mag, mel_len, fname in zip(mag_preds, mel_lens, fnames):
                np.save(os.path.join(fname + '.mag'), mag.cpu().numpy()[:mel_len])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='JSON config files')
    args = parse_with_config(parser)

    global DEVICE
    DEVICE = init_device(args)
    print(f"[{datetime.datetime.now()}] Device: {DEVICE}")

    make_mags(args)

