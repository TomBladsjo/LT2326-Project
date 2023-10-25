import os
import sys
from pathlib import Path
import pandas as pd
import torch
import torchaudio
from torchaudio import transforms
from tqdm.auto import tqdm
import argparse
import math, random

sys.path.insert(0, '../')
import utils
sys.path = sys.path[1:]

def convert_audio(filepath, outdir, name, sample_rate=16000):
        audio = torchaudio.load(filepath)
        audio = utils.resample(audio, sample_rate)
        audio = utils.rechannel(audio, 1)
        # audio = fix_len(audio, duration)
        sgram = utils.spectrogram(audio)
        newpath = os.path.join(outdir,name)+'.pt'
        torch.save(sgram, newpath)

def augment_audio(filepath, outdir, name, sample_rate=16000):
        audio = torchaudio.load(filepath)
        audio = utils.resample(audio, sample_rate)
        audio = utils.rechannel(audio, 1)
        # audio = fix_len(audio, duration)
        audio1 = utils.spec_shift_aug(audio)
        sgram1 = utils.spectrogram(audio1)
        augpath1 = os.path.join(outdir,name)+'_aug1.pt'
        torch.save(sgram1, augpath1)    
        sgram = utils.spectrogram(audio)
        sgram2 = utils.spec_mask_aug(sgram)
        augpath2 = os.path.join(outdir,name)+'_aug2.pt'
        torch.save(sgram2, augpath2)


def preprocess_urdu(basedir, outdir, augment=False):
    print('Preprocessing Urdu...')
    metadata = [['id', 'label']]
    if not Path(outdir).is_dir():
        os.mkdir(outdir)
    directory = os.path.join(basedir, 'data')
    for dirname in tqdm(os.listdir(directory), total=len(os.listdir(directory))):
        dir = os.path.join(directory, dirname)
        if Path(dir).is_dir():
            for file in os.listdir(dir):
                name, extension = os.path.splitext(file)
                if extension == '.wav':
                    metadata.append([name, urdu_labels[dirname]])
                    if augment:
                        metadata.append([name+'_aug1', urdu_labels[dirname]])
                        metadata.append([name+'_aug2', urdu_labels[dirname]])
                        augment_audio(os.path.join(dir, file), outdir, name)
                    convert_audio(os.path.join(dir, file), outdir, name)
    df = utils.make_df(metadata)
    df.to_csv(os.path.join(basedir, 'metadata.csv'), index=False)
            

def preprocess_italian(basedir, outdir, augment=False):
    print('Preprocessing Italian...')
    if not Path(outdir).is_dir():
        os.mkdir(outdir)
    files = []
    directory = os.path.join(basedir, 'data')
    utils.list_paths(directory, files)
    metadata = [['id', 'label']]
    for file in tqdm(files, total=len(files)):
        name, extension = os.path.splitext(os.path.basename(file))
        if extension == '.wav':
            label = name[:3]
            if label in italian_labels:
                metadata.append([name, italian_labels[label]])
                if augment:
                    metadata.append([name+'_aug1', italian_labels[label]])
                    metadata.append([name+'_aug2', italian_labels[label]])
                    augment_audio(file, outdir, name)
                convert_audio(file, outdir, name)
                
    df = utils.make_df(metadata)
    df.to_csv(os.path.join(basedir, 'metadata.csv'), index=False)


def preprocess_estonian(basedir, outdir, augment=False):
    print('Preprocessing Estonian...')
    if not Path(outdir).is_dir():
        os.mkdir(outdir)
    files = []
    directory = os.path.join(basedir, 'data')
    utils.list_paths(directory, files)
    metadata = [['id', 'label']]
    for file in tqdm(files, total=len(files)):
        name, extension = os.path.splitext(os.path.basename(file))
        if extension == '.TextGrid':
            label = get_label_est(file)
            metadata.append([name, estonian_labels[label]])
            audiofile = os.path.splitext(file)[0] + '.wav'
            if augment:
                metadata.append([name+'_aug1', estonian_labels[label]])
                metadata.append([name+'_aug2', estonian_labels[label]])
                augment_audio(audiofile, outdir, name)
            convert_audio(audiofile, outdir, name)
    df = utils.make_df(metadata)
    df.to_csv(os.path.join(basedir, 'metadata.csv'), index=False)
            
def get_label_est(path):
    with open(path) as f:
        lastline = f.readlines()[-1].strip()
        label = lastline.split('"')[-2]
    return label

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess data for emotional speech classification.')
    parser.add_argument("inpath", type=str, help="Path to directory containing subdirectories 'italian', 'urdu' and 'estonian', each with subdirectory 'data' containing the (unzipped) downloaded data in the structure it came.")
    parser.add_argument("-a", "--augment", dest='augment', type=bool, default=False, help="If True, also augments the data.")
    args = parser.parse_args()
    
    urdu_labels = {
        'Angry': 'A',
        'Happy': 'H',
        'Sad': 'S',
        'Neutral': 'N'
    }

    italian_labels = {
        'rab': 'A',
        'gio': 'H',
        'tri': 'S',
        'neu': 'N',
    }

    estonian_labels = {
        'anger': 'A',
        'joy': 'H',
        'sadness': 'S',
        'neutral': 'N',
    }

    if args.augment:
        print('Augmenting data...')

    basicpath = args.inpath   # e.g. '/srv/data/gussodato/emotions'   

    estonian = os.path.join(basicpath, 'estonian')
    preprocess_estonian(estonian, os.path.join(estonian, 'sgrams'), augment=args.augment)
    
    urdu = os.path.join(basicpath, 'urdu')
    preprocess_urdu(urdu, os.path.join(urdu, 'sgrams'), augment=args.augment)

    italian = os.path.join(basicpath, 'italian')
    preprocess_italian(italian, os.path.join(italian, 'sgrams'), augment=args.augment)

    print('Creating train/test splits...')
    for language in ['italian', 'urdu', 'estonian']:
        dir = os.path.join(basicpath, language)
        utils.stratified_divide_data(os.path.join(dir, 'metadata.csv'), dir)

    print('Done!')

    











