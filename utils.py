import pandas as pd
import torch
import torchaudio
from torchaudio import transforms
from tqdm.auto import tqdm
import argparse
import math, random
from pathlib import Path
import os
from sklearn.model_selection import train_test_split

def list_paths(directory, list):
    for path in sorted(Path(directory).iterdir()):
        if path.is_dir():
            list_paths(path, list)
        elif path.is_file():
            list.append(str(os.path.abspath(path)))

def open_audio(file):
    signal, sample_rate = torchaudio.load(file)
    return (signal, sample_rate)

# check if all audio is one channel, otherwise rechannel
def rechannel(audio, new_channel):
    signal, sample_rate = audio
    if (signal.shape[0] == new_channel):
      # Nothing to do
        return audio

    if (new_channel == 1):
      # Convert from stereo to mono by selecting only the first channel
        resig = signal[:1, :]
    else:
      # Convert from mono to stereo by duplicating the first channel
        resig = torch.cat([signal, signal])

    return ((resig, sample_rate))


# check if all have same sample rate otherwise standardize sr
def resample(audio, new_sr):
    signal, sample_rate = audio

    if (sample_rate == new_sr):
      # Nothing to do
        return audio

    num_channels = signal.shape[0]
    # Resample first channel
    resig = torchaudio.transforms.Resample(sample_rate, new_sr)(signal[:1,:])
    if (num_channels > 1):
      # Resample the second channel and merge both channels
        retwo = torchaudio.transforms.Resample(sample_rate, new_sr)(signal[1:,:])
        resig = torch.cat([resig, retwo])

    return ((resig, new_sr))


def fix_len(audio, max_ms): # maxlen in milliseconds
    signal, sample_rate = audio
    num_rows, sig_len = signal.shape
    max_len = sample_rate//1000 * max_ms
    
    if (sig_len > max_len):
    # Truncate the signal to the given length
        signal = signal[:,:max_len]
    
    elif (sig_len < max_len):
    # Length of padding to add at the beginning and end of the signal (padlen before vs after is random)
        pad_begin_len = random.randint(0, max_len - sig_len)
        pad_end_len = max_len - sig_len - pad_begin_len
        # Pad with 0s
        pad_begin = torch.zeros((num_rows, pad_begin_len))
        pad_end = torch.zeros((num_rows, pad_end_len))
        signal = torch.cat((pad_begin, signal, pad_end), 1)
    
    return signal, sample_rate

def spec_shift_aug(aud, shift_limit=0.4):
    sig,sr = aud
    _, sig_len = sig.shape
    shift_amt = int(random.random() * shift_limit * sig_len)
    return (sig.roll(shift_amt), sr)

def spectrogram(audio, n_mels=64, n_fft=1024, hop_len=None):
    signal,sample_rate = audio
    top_db = 80

    # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
    spec = transforms.MelSpectrogram(sample_rate, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(signal)

    # Convert to decibels
    spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
    return (spec) # why in brackets?? test!

def spec_mask_aug(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
    _, n_mels, n_steps = spec.shape
    mask_value = spec.mean()
    aug_spec = spec

    freq_mask_param = max_mask_pct * n_mels
    for _ in range(n_freq_masks):
      aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

    time_mask_param = max_mask_pct * n_steps
    for _ in range(n_time_masks):
      aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

    return aug_spec


def make_df(list_table):
    return pd.DataFrame(list_table[1:], columns=list_table[0])


def stratified_divide_data(csv_file, save_dir, proportion_train=0.8):
    data = pd.read_csv(csv_file)
    train, test = train_test_split(data, train_size=0.8, stratify=data['label'])
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    train.to_csv(os.path.join(save_dir, 'train_data.csv'), index=False)
    test.to_csv(os.path.join(save_dir, 'test_data.csv'), index=False)



