'''
Contains the code for training the vocoder

This code is based off of the train_postnet.py in Transformer-TTS repo.
'''
from utils import *
from preprocess import get_post_dataset, collate_fn_postnet
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
    dataset = get_post_dataset()

    # Model
    model = Vocoder(args.num_mels, args.hidden_size, args.n_fft)
    model = model.to(DEVICE)

    # Optimizer
    optimizer = None
    if args.optim_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim_type == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Continue training
    s_epoch = 0
    if args.load_path is not None:
        if os.path.isfile(args.load_path):
            s_epoch, _, model, optimizer = load_ckp(args.load_path, model, optimizer)
            print(f"[INFO] Training from epoch {s_epoch}.")
        else:
            print(f"[INFO] Could not find checkpoint '{args.load_path}'.")
            print(f"[INFO] Training from initial model.")

    # Scheduler
    scheduler = None
    last_step = ((len(dataset) - args.valid_size) // args.train_batch_size) * s_epoch
    train_steps = ((len(dataset) - args.valid_size) // args.train_batch_size) * args.epochs
    if args.sched_type == "linear":
        scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, train_steps, last_epoch=last_step-1)
    elif args.sched_type == "transformer":
        scheduler = get_transformer_paper_schedule(optimizer, args.warmup_steps, last_epoch=last_step-1)
    elif args.sched_type == "multistep":
        milestones = [i * train_steps for i in args.lr_milestones]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=args.lr_gamma, last_epoch=last_step-1)

    # Loss function
    loss_fn = None
    if args.loss_type == "l2":
        loss_fn = nn.MSELoss(reduction='sum')
    elif args.loss_type == "l1":
        loss_fn = nn.L1Loss(reduction='sum')

    return dataset, s_epoch, model, optimizer, scheduler, loss_fn


def train(args):
    dataset, s_epoch, model, optimizer, scheduler, loss_fn = initialize(args)

    # Make dataloaders
    indices = np.random.permutation(len(dataset))
    valid_dataset = Subset(dataset, indices[-args.valid_size:])
    train_dataset = Subset(dataset, indices[:-args.valid_size])
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn_postnet, drop_last=True, num_workers=args.num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, shuffle=True, collate_fn=collate_fn_postnet, drop_last=False, num_workers=args.num_workers)
    train_epoch_steps = len(train_dataloader)
    valid_epoch_steps = len(valid_dataloader)

    for epoch in range(s_epoch, args.epochs):
        # Train
        model.train()

        pbar = tqdm(train_dataloader)
        pbar.set_description(f"Train Epoch {epoch}")
        losses = []
        for i, data in enumerate(pbar):
            mel, mag = data
            mel = mel.to(DEVICE)
            mag = mag.to(DEVICE)

            mag_pred = model.forward(mel)
            loss = loss_fn(mag_pred, mag)

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            losses.append(loss.detach().cpu().item())

        # Evaluation
        model.eval()
        with torch.no_grad():
            if WRITER:
                sum_params = 0
                for param in model.parameters():
                    sum_params += torch.sum(param).abs().item()
                step = (epoch + 1)*train_epoch_steps
                WRITER.add_scalar('train/vocoder_loss', sum(losses) / len(losses), step)
                WRITER.add_scalar('misc/learning_rate', optimizer.param_groups[0]['lr'], step)
                WRITER.add_scalar("misc/model_params_weight", sum_params, step)

                # Log example
                mel, mag = next(iter(train_dataloader))
                mel = mel.to(DEVICE)
                mag = mag.to(DEVICE)
                mag_pred = model.forward(mel)
                mel_ex = mel[-1].detach().cpu().numpy()
                mag_ex = mag[-1].detach().cpu().numpy()
                pred_ex = mag_pred[-1].detach().cpu().numpy()
                WRITER.add_image('train/mel_input', np.flip(mel_ex.transpose(), axis=0), step, dataformats='HW')
                WRITER.add_image('train/mag_gold', np.flip(mag_ex.transpose(), axis=0), step, dataformats='HW')
                WRITER.add_image('train/mag_pred', np.flip(pred_ex.transpose(), axis=0), step, dataformats='HW')

        losses = []
        pbar = tqdm(valid_dataloader)
        pbar.set_description(f"Valid Epoch {epoch}")
        with torch.no_grad():
            for i, data in enumerate(pbar):
                mel, mag = data
                mel = mel.to(DEVICE)
                mag = mag.to(DEVICE)

                mag_pred = model.forward(mel)
                loss = loss_fn(mag_pred, mag)

                losses.append(loss)

            if WRITER:
                step = (epoch + 1)*valid_epoch_steps
                WRITER.add_scalar('eval/vocoder_loss', sum(losses) / len(losses), step)

                # Log example
                mel, mag = next(iter(valid_dataloader))
                mel = mel.to(DEVICE)
                mag = mag.to(DEVICE)
                mag_pred = model.forward(mel)
                mel_ex = mel[-1].detach().cpu().numpy()
                mag_ex = mag[-1].detach().cpu().numpy()
                pred_ex = mag_pred[-1].detach().cpu().numpy()
                WRITER.add_image('eval/mel_input', np.flip(mel_ex.transpose(), axis=0), step, dataformats='HW')
                WRITER.add_image('eval/mag_gold', np.flip(mag_ex.transpose(), axis=0), step, dataformats='HW')
                WRITER.add_image('eval/mag_pred', np.flip(pred_ex.transpose(), axis=0), step, dataformats='HW')

        # Save model
        if (epoch + 1) % args.save_every == 0:
            save_ckp(epoch, 0, model, optimizer, False, args.checkpoint_path, epoch_save=True)
        save_ckp(epoch, 0, model, optimizer, False, args.checkpoint_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='JSON config files')
    args = parse_with_config(parser)

    global DEVICE
    global WRITER
    DEVICE = init_device(args)
    print(f"[{datetime.datetime.now()}] Device: {DEVICE}")

    if args.tb_log_path:
        WRITER = SummaryWriter(log_dir=args.tb_log_path, flush_secs=60)
        WRITER.add_text("params", str(vars(args)), 0)

    train(args)

    if WRITER:
        WRITER.close()
