from utils import *
from train import (process_batch, supervised_step, discriminator_step, log_loss_metrics, initialize_model)
from preprocess import get_dataset, DataLoader, collate_fn_transformer
from tqdm import tqdm
import argparse
import torch
import numpy as np
from collections import defaultdict
import datetime
import math
import io
import os
import sys

def compute_d_score(outputs, targets):
    outputs = torch.round(outputs)
    targets = torch.round(targets)
    score = torch.sum(outputs == targets)
    return score


def evaluate_test(model, test_dataloader, step, args):
    """
        Expect validation set to have paired speech & text!
        Primary evaluation is PER - can gauge training by the other losses
        We return on 6 other metrics:  autoencoder text loss, autoencoder speech loss,
            ASR loss, and TTS loss, cross model text and cross model speech loss.
    """
    model.eval()
    with torch.no_grad():
        losses = defaultdict(list)
        per, n_iters = 0, 0
        d_score = 0

        for batch, fnames in test_dataloader:
            batch = process_batch(batch)
            x, _ = batch
            text, mel, text_len, mel_len = x

            _, tts_loss = supervised_step(model, batch, args)
            losses['tts'].append(tts_loss.detach().item())

            if args.use_discriminator:
                _, (d_output, d_target) = discriminator_step(model, batch, args)

            text_pred, text_pred_len = model.asr(None, None, mel, mel_len, infer=True)
            _, post_pred, _, stop_lens = model.tts(text, text_len, None, None, infer=True)
            post_pred = post_pred.permute(1, 0, 2).detach().cpu()
            stop_lens = stop_lens.permute(1, 0, 2).detach().cpu()
            for pred, stop_len, fname in zip(post_pred, stop_lens, fnames):
                pred = pred.numpy()
                stop_len = stop_len.item()
                np.save(os.path.join(args.out_mel_dir, fname + '.pt'), pred[:stop_len])
            per += compute_per(text, text_pred.squeeze(-1), text_len, text_pred_len)
            d_score += compute_d_score(d_output, d_target) / args.batch_size
            n_iters += 1

        # TODO: evaluate speech inference somehow?
        compare_outputs(text[-1][:], text_pred[-1][:], text_len[-1], text_pred_len[-1])

    return per/n_iters, losses, d_score/n_iters

def evaluate_wrapper(args):
    set_seed(args.seed)
    print("#### Getting Dataset ####")
    test_dataset = get_dataset('test.csv', ret_file_names=True)
    test_dataloader = DataLoader(test_dataset,
            batch_size=args.eval_batch_size, shuffle=False,
            collate_fn=collate_fn_transformer, drop_last=True,
            num_workers=args.num_workers, pin_memory=True)
    s_epoch, _, model, _, _ = initialize_model(args)
    per, eval_losses, d_score = evaluate_test(model, test_dataloader, -1, args)
    log_loss_metrics(eval_losses, s_epoch, eval=True)
    print("per : {}".format(per))
    print("d_score : {}".format(d_score))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='JSON config files')
    args = parse_with_config(parser)

    global DEVICE
    global WRITER
    DEVICE = init_device(args)
    print(f"[{datetime.datetime.now()}] Device: {DEVICE}")

    evaluate_wrapper(args)

