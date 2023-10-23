import os
import pandas as pd
import torch
import torchaudio
from torchaudio import transforms
from tqdm.auto import tqdm
import argparse
import math, random


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


## TODO data augmentation time shift??

def spectrogram(audio, n_mels=64, n_fft=1024, hop_len=None):
    signal,sample_rate = audio
    top_db = 80

    # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
    spec = transforms.MelSpectrogram(sample_rate, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(signal)

    # Convert to decibels
    spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
    return (spec) 

## TODO data augmentation time and freq masking??


def make_df(list_table):
    return pd.DataFrame(list_table[1:], columns=list_table[0])

def sort_files(path):
    texts = [['id', 'text']]
    audio_paths = [['id', 'path']]
    for p1 in os.listdir(path):
        for p2 in os.listdir(os.path.join(path,p1)):
            for file in os.listdir(os.path.join(path,p1,p2)):
                if os.path.splitext(file)[1] == '.txt':
                    with open(os.path.join(path,p1,p2,file)) as f:
                        for line in f:
                            l = line.split()
                            texts.append([l[0], ' '.join(l[1:])])
                elif os.path.splitext(file)[1] == '.flac':
                    audio_paths.append([os.path.splitext(file)[0], os.path.join(path,p1,p2,file)])
    assert len(texts) == len(audio_paths), 'Number of transcripts does not match number of audio files'   
    return make_df(texts), make_df(audio_paths)


def soundfiles_to_pt(inpath, outpath, transcript_path, get_alphabet=False, get_vocab=False, sample_rate=44100):
        
    print('Finding files...')
    texts, audio_paths = sort_files(inpath)
    if get_alphabet:
        s = set()
        for text in texts['text']:
            s.update(text)
        s = sorted(list(s))
        with open(os.path.join(transcript_path, 'alphabet.txt'), 'w') as f:
            f.writelines(s)
    if get_vocab:
        s = set()
        for text in texts['text']:
            s.update(set(text.split()))
        s = sorted(list(s))
        with open(os.path.join(transcript_path, 'vocab.txt'), 'w') as f:
            for word in s:
                line = [word]+[char for char in word]
                f.write(' '.join(line)+'\n')
    texts.to_csv(os.path.join(transcript_path, 'transcripts.csv'), index=False)
    print('Converting audio...')
    for tuple in tqdm(audio_paths.itertuples(index=False), total=(len(audio_paths))):
        filepath = tuple[1]
        filename = tuple[0]
        audio = torchaudio.load(filepath)
        audio = resample(audio, sample_rate)
        audio = rechannel(audio, 1)
        # audio = fix_len(audio, duration)
        # shift augment?
        sgram = spectrogram(audio)
        # mask augment?

        newpath = os.path.join(outpath,filename)+'.pt'
        torch.save(sgram, newpath)

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess audio data and save as tensors.')
    parser.add_argument("inpath", type=str, help="Path to the audio files.")
    parser.add_argument("outpath", type=str, help="Path to directory to save the output .pt-files in.")
    parser.add_argument("transcript_path", type=str, help="Path to directory to save the transcript file in.")
    parser.add_argument("--sample_rate", "-sr", dest="sample_rate", type=int, default=44100, help="The sample rate for resampling the audio (int, default 44100)")
    parser.add_argument("--alphabet", "-a", dest="alphabet", type=bool, default=False, help="Wether to save the set of characters occurring in transcripts (bool, default=False)")
    parser.add_argument("--vocab", "-v", dest="vocab", type=bool, default=False, help="Wether to save the set of words occurring in transcripts (bool, default=False)")
    args = parser.parse_args()
 
    inpath = args.inpath   
    outpath = args.outpath   
    transcript_path = args.transcript_path
    sample_rate = args.sample_rate

    print('Transforming soundfiles...')
    soundfiles_to_pt(inpath, outpath, transcript_path, get_alphabet=args.alphabet, get_vocab=args.vocab, sample_rate=sample_rate)
    print('Done!')


















    