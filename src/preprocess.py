import audio_parameters as ap
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
import librosa
import numpy as np
from data import raw_text_to_phoneme_ids, data_path
import collections
from scipy import signal
import torch as t
import math


class LJDatasets(Dataset):
    """LJSpeech dataset."""

    def __init__(self, csv_file, root_dir, ret_file_names=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the wavs.

        """
        self.landmarks_frame = pd.read_csv(csv_file, sep='|', header=None)
        self.root_dir = root_dir
        self.ret_file_names = ret_file_names

    def load_wav(self, filename):
        return librosa.load(filename, sr=ap.sr)

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        wav_name = os.path.join(self.root_dir, self.landmarks_frame.loc[idx, 0]) + '.wav'
        fname = wav_name[wav_name.rindex('/') + 1:-4]
        original_text = self.landmarks_frame.loc[idx, 1]

        text = np.asarray(raw_text_to_phoneme_ids(original_text), dtype=np.int32)
        mel = np.load(wav_name[:-4] + '.pt.npy')
        # mel_input = np.concatenate([np.zeros([1,ap.num_mels], np.float32), mel[:-1,:]], axis=0)
        text_length = len(text)
        mel_length = mel.shape[0]
        # pos_text = np.arange(1, text_length + 1)
        # pos_mel = np.arange(1, mel.shape[0] + 1)

        sample = {'text': text, 'mel': mel, 'text_length':text_length, 'mel_length':mel_length}#, 'pos_mel':pos_mel, 'pos_text':pos_text}

        if self.ret_file_names:
            sample['fname'] = fname
        return sample

class PostDatasets(Dataset):
    """LJSpeech dataset."""

    def __init__(self, csv_file, root_dir, is_inf=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the wavs.

        """
        self.landmarks_frame = pd.read_csv(csv_file, sep='|', header=None)
        self.root_dir = root_dir
        self.is_inf = is_inf

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        wav_name = os.path.join(self.root_dir, self.landmarks_frame.loc[idx, 0]) + '.wav'
        fname = wav_name[:-4]
        mel = np.load(wav_name[:-4] + '.pt.npy')
        if self.is_inf:
            return {'mel': mel, 'fname': fname}
        else:
            mag = np.load(wav_name[:-4] + '.mag.npy')
        sample = {'mel':mel, 'mag':mag}

        return sample

def collate_fn_transformer(batch):

    # Puts each data field into a tensor with outer dimension batch size
    if isinstance(batch[0], collections.Mapping):

        text = [d['text'] for d in batch]
        mel = [d['mel'] for d in batch]
        # mel_input = [d['mel_input'] for d in batch]
        mel_length = [d['mel_length'] for d in batch]
        text_length = [d['text_length'] for d in batch]
        # pos_mel = [d['pos_mel'] for d in batch]
        # pos_text= [d['pos_text'] for d in batch]
        if 'fname' in batch[0]:
            fnames = [d['fname'] for d in batch]
            fnames = [i for i, _ in sorted(zip(fnames, text_length), key=lambda x: x[1], reverse=True)]

        text = [i for i,_ in sorted(zip(text, text_length), key=lambda x: x[1], reverse=True)]
        mel = [i for i, _ in sorted(zip(mel, text_length), key=lambda x: x[1], reverse=True)]
        mel_length = [i for i, _ in sorted(zip(mel_length, text_length), key=lambda x: x[1], reverse=True)]
        # mel_input = [i for i, _ in sorted(zip(mel_input, text_length), key=lambda x: x[1], reverse=True)]
        # pos_text = [i for i, _ in sorted(zip(pos_text, text_length), key=lambda x: x[1], reverse=True)]
        # pos_mel = [i for i, _ in sorted(zip(pos_mel, text_length), key=lambda x: x[1], reverse=True)]
        text_length = sorted(text_length, reverse=True)
        # PAD sequences with largest length of the batch
        text = _prepare_data(text).astype(np.int32)
        mel = _pad_mel(mel)
        # mel_input = _pad_mel(mel_input)
        # pos_mel = _prepare_data(pos_mel).astype(np.int32)
        # pos_text = _prepare_data(pos_text).astype(np.int32)

        # return t.LongTensor(text), t.FloatTensor(mel), t.FloatTensor(mel_input), t.LongTensor(pos_text), t.LongTensor(pos_mel), t.LongTensor(text_length)
        if 'fname' in batch[0]:
            return (t.as_tensor(text, dtype=t.long), t.as_tensor(mel, dtype=t.float), \
                t.as_tensor(text_length, dtype=t.long), t.as_tensor(mel_length, dtype=t.long)), fnames

        return t.as_tensor(text, dtype=t.long), t.as_tensor(mel, dtype=t.float), \
            t.as_tensor(text_length, dtype=t.long), t.as_tensor(mel_length, dtype=t.long)

    raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}"
                     .format(type(batch[0]))))

def collate_fn_postnet(batch):

    # Puts each data field into a tensor with outer dimension batch size
    if isinstance(batch[0], collections.Mapping):

        mel = [d['mel'] for d in batch]
        mel_lens = [len(m) for m in mel]

        # PAD sequences with largest length of the batch
        mel = _pad_mel(mel)

        if 'mag' in batch[0]:
            mag = [d['mag'] for d in batch]
            mag = _pad_mel(mag)
            return t.as_tensor(mel, dtype=t.float), t.as_tensor(mag, dtype=t.float)
        elif 'fname' in batch[0]:
            fnames = [d['fname'] for d in batch]
            return t.as_tensor(mel, dtype=t.float), mel_lens, fnames
        return t.as_tensor(mel, dtype=t.float)

    raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}"
                     .format(type(batch[0]))))

def _pad_data(x, length):
    _pad = 0
    return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=_pad)

def _prepare_data(inputs):
    max_len = max((len(x) for x in inputs))
    return np.stack([_pad_data(x, max_len) for x in inputs])

def _pad_per_step(inputs):
    timesteps = inputs.shape[-1]
    return np.pad(inputs, [[0,0],[0,0],[0, ap.outputs_per_step - (timesteps % ap.outputs_per_step)]], mode='constant', constant_values=0.0)

def get_param_size(model):
    params = 0
    for p in model.parameters():
        tmp = 1
        for x in p.size():
            tmp *= x
        params += tmp
    return params

def get_dataset(split_file, ret_file_names=False):
    return LJDatasets(os.path.join(data_path,split_file), os.path.join(data_path,'wavs'), ret_file_names=ret_file_names)

def get_post_dataset():
    return PostDatasets(os.path.join(data_path,'metadata.csv'), os.path.join(data_path,'wavs'))

def get_test_mel_dataset(mels_dir, audio_list_file):
    return PostDatasets(audio_list_file, mels_dir, is_inf=True)

def _pad_mel(inputs):
    _pad = 0
    def _pad_one(x, max_len):
        mel_len = x.shape[0]
        return np.pad(x, [[0,max_len - mel_len],[0,0]], mode='constant', constant_values=_pad)
    max_len = max((x.shape[0] for x in inputs))
    return np.stack([_pad_one(x, max_len) for x in inputs])

