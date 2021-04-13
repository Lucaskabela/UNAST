'''
Contains any and all code we didnt want to put somewhere else
'''
import torch
import numpy as np 
import random
import librosa
import audio_parameters as ap
import json
import sys
import torch.utils.tensorboard as tb
import shutil
from jiwer import wer

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2


def compute_per(ground_truth, hypothesis, ground_truth_lengths, hypothesis_lengths):
    # Given two tensors of size [batch_size x sent_len], compute the PER!
    # lengths should be a tensor of [batch_size], used for ignoring padding

    gt_sents = []
    hyp_sents = []
    for b in range(ground_truth.shape[0]):
        gt_sents.append(' '.join(ground_truth[b][:].tolist()[:ground_truth_lengths[b]]))
        hyp_sents.append(' '.join(hypothesis[b][:].tolist()[:hypothesis_lengths[b]]))
    return wer(gt_sents, hyp_sents)


def noise_fn(to_noise, mask_p=.3, swap_p=0):
    """
    to_noise should be [batch x seq_len x dim], and we want to hide entire swaths
    of the sequence
    """
    # NOTE: swap_p does nothing!
    gen = torch.zeros((to_noise.shape[0], to_noise.shape[1]), device=to_noise.device)
    gen.fill_(1-mask_p)
    zero_mask = torch.bernoulli(gen).unsqueeze(-1)
    return to_noise * zero_mask


def sent_lens_to_mask(lens, max_length):
    """
    lens should be tensor of dim [batch_size]
    """
    return torch.from_numpy(np.asarray([[1 if j < lens.data[i].item() else 0 for j in range(0, max_length)] for i in range(0, lens.shape[0])]), 
        device=lens.device)

def set_seed(seed):
    '''
    Sets torch, numpy, and random library with seed for reproducibility
    See: https://pytorch.org/docs/stable/notes/randomness.html for more details
    on setting determinism

    Args:
        - seed: An integer seed for consistency in different runs
    ''' 
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def init_device(args):
    if torch.cuda.is_available() and args.gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def init_logger(log_dir=None):
    train_logger, valid_logger = None, None
    if log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(log_dir, "train"))
        valid_logger = tb.SummaryWriter(path.join(log_dir, "valid"))
    return train_logger, valid_logger


# Next two methods courtesy of: https://towardsdatascience.com/how-to-save-and-load-a-model-in-pytorch-with-a-complete-example-c2920e617dee
def save_ckp(epoch, valid_loss, model, optimizer, is_best, checkpoint_path):
    """
    state: checkpoint we want to save.  State is a dict with keys:
            ['epoch','valid_loss_min', 'state_dict', 'optimizer']
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    state = {
        'epoch': epoch + 1,
        'valid_loss_min': valid_loss,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    f_path = checkpoint_path + '/model_{}.ckpt'.format(state['epoch'])
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = checkpoint_path + '/model_best.ckpt'
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)


def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)

    # initialize state_dicts from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


    return checkpoint['epoch'], checkpoint['valid_loss_min'], model, optimizer


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
