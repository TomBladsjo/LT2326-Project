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
    resig = torchaudio.transforms.Resample(sample_rate, newsr)(signal[:1,:])
    if (num_channels > 1):
      # Resample the second channel and merge both channels
        retwo = torchaudio.transforms.Resample(sample_rate, newsr)(signal[1:,:])
        resig = torch.cat([resig, retwo])

    return ((resig, newsr))


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
    return (spec) # why in brackets?? test!

## TODO data augmentation time and freq masking??





def wav_to_pt(inpath, outpath, metadata_csv, column_name, duration=10000, sample_rate=44100):
    if inpath[-1] != '/':
        inpath = inpath + '/'
    if outpath[-1] != '/':
        outpath = outpath + '/'
    metadata = pd.read_csv(metadata_csv)
    for i in tqdm(range(len(metadata[column_name]))):
        filename = metadata[column_name][i]
        file = inpath+filename if 'wav' in filename[-4:] else inpath+filename+'.wav'
        audio = torchaudio.load(file)
        audio = resample(audio, sample_rate)
        audio = rechannel(audio, 1)
        audio = fix_len(audio, duration)
        sgram = spectrogram(audio)

        newname = outpath+filename+'.pt'
        torch.save(sgram, newname)

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess audio data and save as tensors (adapted specifically for the warblr10k dataset).')
    parser.add_argument("inpath", type=str, help="Path to the audio file directory.")
    parser.add_argument("outpath", type=str, help="Path to directory to save the output sgram files")
    parser.add_argument("metadata", type=str, help="Metadata file (csv)")
    parser.add_argument("--column_id", "-c", dest="column_id", type=str, default="itemid", help="The ID of the metadata column containing the file names")
    parser.add_argument("--duration", "-d", dest="duration", type=int, default=10000, help="The duration of the output audioclips in milliseconds (int, default 10000)")
    parser.add_argument("--sample_rate", "-sr", dest="sample_rate", type=int, default=44100, help="The sample rate for resampling the audio (int, default 44100)")
    args = parser.parse_args()
 
    inpath = args.inpath # e.g. '/srv/data/gussodato/SONYC-UST/audio/'
    outpath = args.outpath  # e.g. '/srv/data/gussodato/SONYC-UST/sgrams'
    metadata = args.metadata  # e.g. '/srv/data/gussodato/SONYC-UST/metadata/annotations.csv'
    column_id = args.column_id
    duration = args.duration
    sample_rate = args.sample_rate

    print('Transforming wav-files...')
    wav_to_pt(inpath, outpath, metadata, column_id, duration, sample_rate)
    print('Done!')


















    