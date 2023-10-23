import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from IPython import embed
from tqdm.auto import tqdm 
import matplotlib.pyplot as plt
import random
import csv



class ASRDataset(Dataset):
    def __init__(self, transcript_csv, indata_dir, converter, lower=True):
        metadata = pd.read_csv(transcript_csv)
        self.files = metadata['id']
        self.transcripts = metadata['text']
        self.data_dir = indata_dir if indata_dir[-1] == '/' else indata_dir+'/'
        self.lower = lower
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        
        file = os.path.join(self.data_dir,self.files[idx])+'.pt'
        x = torch.load(file)
        if self.lower:
            y = [converter[char] for char in self.transcripts[idx].lower()]
        else:
            y = [converter[char] for char in self.transcripts[idx]]
  
        return {'input': x, 'transcript': torch.tensor(y, dtype=torch.long)}


class CharacterConverter():
    def __init__(self, alphabet, lower=True):
        if lower:
            alphabet = alphabet.lower()
        self.char2idx = {alphabet[i]: i+1 for i in range(len(alphabet))}
        self.char2idx[''] = 0
        self.idx2char = {v: k for k,v in self.char2idx.items()}
        assert len(self.char2idx) == len(self.idx2char)
        self.blankindex = self.char2idx['']
        
    def __len__(self):
        return len(self.idx2char)

    def __getitem__(self, idx):
        assert (type(idx) == str or type(idx) == int), 'idx must be either str or int'
        if type(idx) == str:
            return self.char2idx[idx]
        elif type(idx) == int:
            return self.idx2char[idx]


class ASRCollator():
    def __call__(self, batch):
        # each item will have shape (channels, n_mels, time), where time is the only one that varies
            width = [item['input'].shape[2] for item in batch]
            C, H = batch[0]['input'].shape[:2]
            inputs = torch.ones([len(batch), C, H, 
                               max(width)], dtype=torch.float32)
            for idx, item in enumerate(batch):
                try:
                    inputs[idx, :, :, 0:item['input'].shape[2]] = item['input']
                except:
                    print(inputs.shape)
            item = {'input': inputs}
            if 'transcript' in batch[0].keys():
                texts = [item['transcript'] for item in batch]
                item['transcript'] = texts
            return item



class ASRModel(nn.Module):
    def __init__(self, options):
        super(ASRModel, self).__init__()

        # Layers:

        # CNN part
        cnn = nn.Sequential()
        # 1st conv-relu-pool
        self.conv1 = nn.Conv2d(in_channels=options['in_channels'], out_channels=options['cnn_channels']*2,
			kernel_size=options['kernel_size'])
        self.relu1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool2d(kernel_size=options['pool_size'])

        # 2nd conv-relu-pool
        self.conv2 = nn.Conv2d(in_channels=options['cnn_channels']*2, out_channels=options['cnn_channels']*4,
			kernel_size=options['kernel_size'])
        self.relu2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool2d(kernel_size=options['pool_size'])

        # 3rd conv-relu-pool with adaptive average pooling for fc input normalizing
        self.conv3 = nn.Conv2d(in_channels=options['cnn_channels']*4, out_channels=options['cnn_channels']*8,
			kernel_size=options['kernel_size'])
        self.relu3 = nn.ReLU()
        H, W = 1, options['avg_pool_out']
        self.avgpool3 = nn.AvgPool2d(kernel_size=options['pool_size']) 
      
        # fc-relu
        C = options['cnn_channels']*8  # == number of features
        self.fc1 = nn.Linear(C*4, options['fc1_out']) # takes (batch,width,features) and returns (b, w, fc1_out)
        self.relu4 = nn.ReLU()
       
        # RNN part  (takes (sequence length, batch, input size), equal to (W, B, fc1_out))
        self.lstm = nn.LSTM(input_size=options['fc1_out'], hidden_size=options['rnn_hidden'], num_layers=4, bidirectional=True)
        # (outputs (W, B, D*hidden_size))
        # Loss function wants (sequence_length, batch, n_classes)
        self.classifier = nn.Linear(options['rnn_hidden']*2, options['n_classes'])
        self.logsoft = nn.LogSoftmax(dim=2) # (2 is the dimension of the classes)

    
    def forward(self, x):

        # CNN part:
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.avgpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.avgpool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.avgpool3(x)

        # Reshape for input to linear:
        B, C, H, W = x.shape
        x = x.view(B, W, H*C)

        # FC layer:
        x = self.fc1(x)
        x = self.relu4(x)

        # Reshape for input to LSTM (because batch_first=False):
        x = torch.transpose(x, 0, 1)

        # RNN:
        x, _ = self.lstm(x)
        x = self.classifier(x)
        x = self.logsoft(x)
           
        return x


def train_asr(model, options):
    device = options['device']
    loss_fn = nn.CTCLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=options['lr'])
    logfile = os.path.join(options['plot_dir'], 'training_log')
    print('Training model...\n')
    train_loader = DataLoader(options['train_data'], batch_size=options['batch_size'], shuffle=True, collate_fn=ASRCollator())
    val_loader = DataLoader(options['val_data'], batch_size=options['batch_size'], shuffle=False, collate_fn=ASRCollator())

    train_losses, eval_losses = [], []
    for epoch in range(options['epochs']):
        print(f'Training epoch {epoch}...')
        model.to(device)
        model.train()
        ep_train_losses, ep_eval_losses = [], []
        for i, batch in enumerate(tqdm(train_loader)):
            input, targets = batch['input'], batch['transcript']
            target_lengths = torch.tensor([len(sequence) for sequence in targets], dtype=torch.long)
            target = torch.cat(targets)
            output = model(input.to(device))
            T, B, C = output.shape
            output_lengths = torch.tensor([T]*B, dtype=torch.long)
            loss = loss_fn(output, target, output_lengths, target_lengths)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            ep_train_losses.append(float(loss))

        print('Evaluating...')
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_loader)):
                input, targets = batch['input'], batch['transcript']
                target_lengths = torch.tensor([len(sequence) for sequence in targets], dtype=torch.long)
                target = torch.cat(targets)
                output = model(input.to(device))
                T, B, C = output.shape
                output_lengths = torch.tensor([T]*B, dtype=torch.long)
                loss = loss_fn(output, target, output_lengths, target_lengths)
                ep_eval_losses.append(float(loss))

        assert len(ep_train_losses) and len(ep_eval_losses), 'oops, no losses in the list'
        tot_ep_train_loss = round(sum(ep_train_losses)/len(ep_train_losses), 4) 
        tot_ep_eval_loss = round(sum(ep_eval_losses)/len(ep_eval_losses), 4)

        log = [tot_ep_train_loss, tot_ep_eval_loss]
        with open(logfile, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(log)
        
        print(f'Epoch {epoch}:\nTraining loss: {tot_ep_train_loss}\nValidation loss: {tot_ep_eval_loss}')
        train_losses.append(tot_ep_train_loss)
        eval_losses.append(tot_ep_eval_loss)

        if eval_losses[-1] == min(eval_losses):
            torch.save(model.cpu().state_dict(), os.path.join(options['checkpoint_dir'], 'best_model.pt'))

        if epoch > options['min_epochs'] and options['early_stopping']:
            if stopping_criterion(eval_losses, patience=options['patience']):
                print(f'Early stopping after {options["patience"]} epochs without improvement on validation set.')
                return train_losses, eval_losses
    return train_losses, eval_losses

        

def stopping_criterion(eval_losses, patience=10):
    window = eval_losses[-(patience+1):]
    stop = True
    for i in range(len(window)-1):
        if window[i] > window[i+1]:
            stop = False
    return stop
    
    

def div_data(dataset, proportion_train):
    n_items = len(dataset)
    n_train = round(n_items * proportion_train)
    n_val = n_items - n_train
    train_data, val_data = random_split(dataset, [n_train, n_val])
    return train_data, val_data


def plot_losses(train_loss, val_loss, save_dir):

    fig, ax = plt.subplots()
    plt.plot(range(len(train_loss)), train_loss, color='orange', label='Training loss')
    plt.plot(range(len(val_loss)), val_loss, color='blue', label='Validation loss')
    plt.legend()
    plt.xlabel("Number of epochs")
    plt.title("Loss")
    plt.savefig(os.path.join(save_dir,'loss_plot'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ASR model.')
    parser.add_argument("traindata", type=str, help="Path to the training input files.")
    parser.add_argument("train_transcript", type=str, help="File (csv) containing training input file IDs and transcriptions.")
    parser.add_argument("devdata", type=str, help="Path to the development set input files.")
    parser.add_argument("dev_transcript", type=str, help="File (csv) containing devset input file IDs and transcriptions.")
    parser.add_argument("-a", "--alphabet", dest='alphabet', default=None, help="File (txt) containing the permitted characters.")
    parser.add_argument("-d", "--device", dest='device', default='cuda:0', help="GPU on which to perform computations.")
    args = parser.parse_args()

    if args.alphabet == None:
        alphabet = [''] + list("abcdefghijklmnopqrstuvwxyz' ")
    else:
        alphabet_file = args.alphabet  #'/srv/data/gussodato/LibriSpeech/metadata/dev/alphabet.txt'
        with open(alphabet_file) as f:
            alphabet = f.read()
            
    converter = CharacterConverter(alphabet)

    train_data = ASRDataset(args.train_transcript, args.traindata, converter)
    dev_data = ASRDataset(args.dev_transcript, args.devdata, converter)
            
    # indata_dir = args.traindata  # '/srv/data/gussodato/LibriSpeech/sgrams/dev/'
    # transcript = args.train_transcript  #'/srv/data/gussodato/LibriSpeech/metadata/dev/transcripts.csv'
        
    model_options = {
        'in_channels': 1,
        'n_classes': len(converter),
        'cnn_channels': 64,
        'rnn_hidden': 256,
        'avg_pool_out': 800,  # this is what decides the max output sequence length! make sure there is some extra space here just in case!
        'fc1_out': 256,
        'kernel_size': 5,
        'pool_size': 2,
        'conv_stride': 1
    }

    # dataset = ASRDataset(transcript, indata_dir, converter)
    # train, eval = div_data(dataset, 0.8)

    model = ASRModel(model_options)
    
    training_options = {
        'train_data': train_data,
        'val_data': dev_data,
        'device': torch.device(args.device),
        'epochs': 200,
        'batch_size': 2,
        'lr': 0.001,
        'early_stopping': True,
        'min_epochs': 10,
        'patience': 5,
        'checkpoint_dir': './checkpoints/',
        'plot_dir': './plots/'
    }

    train_losses, eval_losses = train_asr(model, training_options)
    plot_losses(train_losses, eval_losses, training_options['plot_dir'])













