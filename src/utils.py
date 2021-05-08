'''
Contains any and all code we didnt want to put somewhere else
'''
import torch
import numpy as np
import copy
import random
import librosa
import audio_parameters as ap
import json
import sys
import torch.utils.tensorboard as tb
import shutil
from jiwer import wer
from scipy import signal
from data import sequence_to_text
import os

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2


def compute_per(ground_truth, hypothesis, ground_truth_lengths, hypothesis_lengths):
    # Given two tensors of size [batch_size x sent_len], compute the PER!
    # lengths should be a tensor of [batch_size], used for ignoring padding

    gt_sents = []
    hyp_sents = []
    for b in range(ground_truth.shape[0]):
        gt_sents.append(' '.join(map(str, ground_truth[b][:].tolist()[:ground_truth_lengths[b]])))
        hyp_sents.append(' '.join(map(str, hypothesis[b][:].tolist()[:hypothesis_lengths[b]])))

    return wer(gt_sents, hyp_sents)

def compare_outputs(ground_truth, hypothesis, gt_len, hyp_len):
    print(f'Model prediction of length {hyp_len} ', sequence_to_text(hypothesis.tolist()[:hyp_len]))
    print(f'Ground Truth of length {gt_len} ', sequence_to_text(ground_truth.tolist()[:gt_len]))

def noise_fn(to_noise, mask_p=.3, swap_p=0):
    """
    to_noise should be [batch x seq_len x dim], and we operate on timesteps
    of the sequence
    """
    # NOTE: swap_p does nothing as of right now!
    gen = torch.zeros((to_noise.shape[0], to_noise.shape[1]), device=to_noise.device)
    gen.fill_(1-mask_p)
    zero_mask = torch.bernoulli(gen).unsqueeze(-1)
    return to_noise * zero_mask

def specaugment(mel, mel_len, freq_mask=20, time_mask=100, replace_with_zero=False):
    # No timewarp because I don't hate myself that much
    with torch.no_grad():
        res = mel.detach().clone()
        freq_len = torch.randint(0, freq_mask, mel_len.shape)
        time_len = torch.randint(0, time_mask, mel_len.shape)

        for i in range(freq_len.shape[0]):
            mean_ = res[i].mean()
            f = freq_len[i].item()
            t = time_len[i].item()
            # Make sure mel_len[i] - t > 0!!
            if mel_len[i] - t <= 0:
                # new t should just be less than half the length, why not
                t = random.randrange(0, mel_len[i] // 2)
            f_zero = random.randrange(0, mel_len[i]- f)
            t_zero = random.randrange(0, mel_len[i]- t)
            if replace_with_zero:
                res[i][:][f_zero:f_zero+f] = 0
                res[i][t_zero:t_zero+t][:] = 0
            else:
                res[i][:][f_zero:f_zero+f] = mean_
                res[i][t_zero:t_zero+t][:] = mean_

    return res

def sent_lens_to_mask(lens, max_length):
    """
    lens should be tensor of dim [batch_size]
    """
    m = [[1 if j < lens.data[i].item() else 0 for j in range(0, max_length)]
        for i in range(0, lens.shape[0])]
    return torch.as_tensor(m, device=lens.device, dtype=torch.bool)

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
    if torch.cuda.is_available() and args.use_gpu:
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

class TeacherRatio():
    def __init__(self, args):
        self.iter = 0
        self.val = args.teacher_init_val
        self.gamma = args.teacher_gamma
        self.start_step = args.teacher_decay_start
        self.stop_step = args.teacher_decay_end

    def step(self):
        self.iter += 1

    def get_val(self):
        # Do not change val in case user loads w/iter
        if self.start_step <= self.iter:
            power = min(self.iter, self.stop_step) - self.start_step
            return self.val * (self.gamma ** power)
        else:
            return self.val

def get_teacher_ratio(args):
    return TeacherRatio(args)

# Next two methods courtesy of: https://towardsdatascience.com/how-to-save-and-load-a-model-in-pytorch-with-a-complete-example-c2920e617dee
def save_ckp(epoch, valid_loss, model, optimizer, is_best, checkpoint_path, temporary_save=False, epoch_save=False):
    """
    state: checkpoint we want to save.  State is a dict with keys:
            ['epoch','valid_loss_min', 'state_dict', 'optimizer']
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    state = {
        'epoch': epoch + 1,
        'valid_loss_min': valid_loss,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    # save checkpoint data to the path given, checkpoint_path
    if temporary_save:
        f_path = checkpoint_path + '/model_temporary.ckpt'
        torch.save(state, f_path)
        return

    if epoch_save:
        f_path = checkpoint_path + f'/model_{epoch}.ckpt'
        torch.save(state, f_path)
        return

    f_path = checkpoint_path + '/model_most_recent.ckpt'
    torch.save(state, f_path)

    # if it is a best model, min validation loss
    if is_best:
        best_fpath = checkpoint_path + '/model_best.ckpt'
        # copy that checkpoint file to best path given, best_model_path
        torch.save(state, best_fpath)


def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into
    optimizer: optimizer we defined in previous training
    """
    if not os.path.exists(checkpoint_fpath):
        raise Exception("There is no model at the desired checkpoint")

    # load check point
    checkpoint = torch.load(checkpoint_fpath, map_location='cuda:0')

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


###### CODE BELOW ARE TAKEN FROM Transformer-TTS ######


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


def spectrogram2wav(mag):
    '''# Generate wave file from linear magnitude spectrogram using Griffin-Lim
    Args:
      mag: A numpy array of (T, 1+n_fft//2)
    Returns:
      wav: A 1-D numpy array.
    '''
    # transpose
    mag = mag.T

    # de-noramlize
    mag = (np.clip(mag, 0, 1) * ap.max_db) - ap.max_db + ap.ref_db

    # to amplitude
    mag = np.power(10.0, mag * 0.05)

    # wav reconstruction
    wav = griffin_lim(mag**ap.power)

    # de-preemphasis
    wav = signal.lfilter([1], [1, -ap.preemphasis], wav)

    # trim
    wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)


def griffin_lim(spectrogram):
    '''Applies Griffin-Lim's raw.'''
    X_best = copy.deepcopy(spectrogram)
    for i in range(ap.n_iter):
        X_t = invert_spectrogram(X_best)
        est = librosa.stft(X_t, ap.n_fft, ap.hop_length, win_length=ap.win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = invert_spectrogram(X_best)
    y = np.real(X_t)

    return y


def invert_spectrogram(spectrogram):
    '''Applies inverse fft.
    Args:
      spectrogram: [1+n_fft//2, t]
    '''
    return librosa.istft(spectrogram, ap.hop_length, win_length=ap.win_length, window="hann")
