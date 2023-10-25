import os
from pathlib import Path
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import csv



class LabelIndexer():
    def __init__(self, labels):
        
        self.label2idx = {labels[i]: i for i in range(len(labels))}
        self.idx2label = {v: k for k,v in self.label2idx.items()}
        assert len(self.label2idx) == len(self.idx2label)
        
    def __len__(self):
        return len(self.idx2label)
        
    def __getitem__(self, idx):
        assert (type(idx) == str or type(idx) == int), 'index must be either str or int'
        if type(idx) == str:
            return self.label2idx[idx]
        elif type(idx) == int:
            return self.idx2label[idx]
            

class EmoDataset(Dataset):
    def __init__(self, metadata_csv, indata_dir, converter, lower=True):
        if type(metadata_csv)==str:
            metadata = pd.read_csv(metadata_csv)
        else:
            metadata = metadata_csv
        self.files = metadata['id']
        self.labels = metadata['label']
        self.data_dir = indata_dir if indata_dir[-1] == '/' else indata_dir+'/'
        self.converter = converter
        self.classnames = list(converter.label2idx.keys())
        self.proportions = metadata.value_counts('label', normalize=True)
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        
        file = os.path.join(self.data_dir,str(self.files[idx]))+'.pt'
        x = torch.load(file)
        y = self.converter[self.labels[idx]]
  
        return {'input': x, 'label': torch.tensor(y, dtype=torch.long)}



class EmoCollator():
    def __call__(self, batch):
        # each item will have shape (channels, n_mels, time), where time is the only one that varies
            width = max([item['input'].shape[2] for item in batch])
            W = max(width, 10)
            C, H = batch[0]['input'].shape[:2]
            inputs = torch.ones([len(batch), C, H, 
                               W], dtype=torch.float32)
            for idx, item in enumerate(batch):
                try:
                    inputs[idx, :, :, 0:item['input'].shape[2]] = item['input']
                except:
                    print(inputs.shape)
            item = {'input': inputs}
            if 'label' in batch[0].keys():
                labels = [item['label'] for item in batch]
                item['label'] = torch.tensor(labels, dtype=torch.long)
            return item


class EmoClassModel(nn.Module):
    def __init__(self, options):
        super(EmoClassModel, self).__init__()

        # Layers:

        # CNN part
        cnn = nn.Sequential()
        # 1st conv-relu-pool
        self.conv1 = nn.Conv2d(in_channels=options['in_channels'], out_channels=options['cnn_channels']*2,
			kernel_size=options['kernel_size'])
        self.relu1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool2d(kernel_size=options['pool_size'])

        # 2nd conv-relu-pool with adaptive average pooling for fc input normalizing
        self.conv2 = nn.Conv2d(in_channels=options['cnn_channels']*2, out_channels=options['cnn_channels']*4,
			kernel_size=options['kernel_size'])
        self.relu2 = nn.ReLU()
        H, W = 1, options['avg_pool_out']
        self.avgpool2 = nn.AdaptiveAvgPool2d((H,W))  # will return (batch, channels, H, W) with H being 1
      
        # fc-tanh
        C = options['cnn_channels']*4  # == number of features
        self.fc1 = nn.Linear(C, options['fc1_out']) # takes (batch,width,features) and returns (b, w, fc1_out)
        self.tanh = nn.Tanh()
       
        # RNN part  (takes (batch, sequence length, input size), equal to (B, W, fc1_out))
        self.lstm = nn.LSTM(input_size=options['fc1_out'], hidden_size=options['rnn_hidden'], num_layers=2, batch_first=True, bidirectional=True)
        # outputs (B, W, D*hidden_size), (D*num_layers, B, hidden_size)(concatenation of final hidden states forward and backward)

        self.classifier = nn.Linear(options['rnn_hidden']*W*2, options['n_classes'])
       

    
    def forward(self, x):

        # CNN part:
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.avgpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.avgpool2(x)

        # Reshape for input to linear:
        B, C, H, W = x.shape
        x = x.view(B, W, H*C)

        # FC layers:
        x = self.fc1(x)
        x = self.tanh(x)

        # RNN:
        out, (hidden, cell) = self.lstm(x)

        # classifier
        x = torch.flatten(out, 1)
        x = self.classifier(x)      
        
        return x


def train_emotion_model(options):
    model = options['model']
    if not Path(options['plot_dir']).is_dir():
        os.mkdir(options['plot_dir'])
    if not Path(options['checkpoint_dir']).is_dir():
        os.mkdir(options['checkpoint_dir'])
    logfile = os.path.join(options['plot_dir'], options['experiment_name']+'_training_log')
    with open(logfile, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['training_loss', 'validation_loss', 'training_accuracy', 'validation_accuracy'])
    device = options['device']
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=options['lr'])
    print(f'\nTraining {options["experiment_name"]} model')
    train_loader = DataLoader(options['train_data'], batch_size=options['batch_size'], shuffle=True, collate_fn=EmoCollator())
    val_loader = DataLoader(options['val_data'], batch_size=options['batch_size'], shuffle=False, collate_fn=EmoCollator())

    train_losses, eval_losses = [], []
    train_acc, eval_acc = [], []
    for epoch in range(options['epochs']):
        print(f'\nTraining epoch {epoch}...\n')
        model.to(device)
        model.train()
        ep_train_losses, ep_eval_losses = [], []
        ep_train_acc, ep_eval_acc = [], []
        for i, batch in enumerate(tqdm(train_loader)):
            try:
                x, y_true = batch['input'], batch['label']
                y_pred = model(x.to(device))
                loss = loss_fn(y_pred, y_true.to(device))
                # get accuracy
                batch_preds = [torch.argmax(prediction) for prediction in y_pred.cpu()]
                assert len(y_true) == len(batch_preds)
                batch_accuracy = accuracy_score(y_true.cpu(), batch_preds)
                ep_train_acc.append(batch_accuracy)
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                ep_train_losses.append(float(loss))
            except:
                print('Something went wrong here, skipping batch!') # (bc something is up with the Estonian dataset and I couldn't find out what)

        print('Evaluating...')
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_loader)):
                try:
                    x, y_true = batch['input'], batch['label']
                    y_pred = model(x.to(device))
                    loss = loss_fn(y_pred, y_true.to(device))
                    ep_eval_losses.append(float(loss))
                    # get accuracy
                    batch_preds = [torch.argmax(prediction) for prediction in y_pred.cpu()]
                    assert len(y_true) == len(batch_preds)
                    batch_accuracy = accuracy_score(y_true.cpu(), batch_preds)
                    ep_eval_acc.append(batch_accuracy)
                except:
                    print('Something went wrong here, skipping batch!')

        assert len(ep_train_losses) and len(ep_eval_losses), 'oops, no losses in the list'
        assert len(ep_train_acc) and len(ep_eval_acc), 'oops, no accuracies in the list'
        avg_ep_train_loss = round(sum(ep_train_losses)/len(ep_train_losses), 4) 
        avg_ep_eval_loss = round(sum(ep_eval_losses)/len(ep_eval_losses), 4)
        avg_ep_train_acc = round(sum(ep_train_acc)/len(ep_train_acc), 4) 
        avg_ep_eval_acc = round(sum(ep_eval_acc)/len(ep_eval_acc), 4)

        log = [avg_ep_train_loss, avg_ep_eval_loss, avg_ep_train_acc, avg_ep_eval_acc]
        with open(logfile, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(log)
        
        print(f'{options["experiment_name"]}, epoch {epoch}:\nTraining loss: {avg_ep_train_loss}\nTraining accuracy: {avg_ep_train_acc}')
        print(f'Validation loss: {avg_ep_eval_loss}\nValidation accuracy: {avg_ep_eval_acc}')
        train_losses.append(avg_ep_train_loss)
        eval_losses.append(avg_ep_eval_loss)
        train_acc.append(avg_ep_train_acc)
        eval_acc.append(avg_ep_eval_acc)

        if eval_losses[-1] == min(eval_losses):
            torch.save(model.cpu().state_dict(), os.path.join(options['checkpoint_dir'], options['experiment_name']+'_best_model.pt'))

        if epoch > options['min_epochs'] and options['early_stopping']:
            if stopping_criterion(eval_losses, patience=options['patience']):
                print(f'Early stopping after {options["patience"]} epochs without improvement on validation set.')
                return train_losses, eval_losses, train_acc, eval_acc
    return train_losses, eval_losses, train_acc, eval_acc

        

def stopping_criterion(eval_losses, patience=10):
    window = eval_losses[-(patience+1):]
    stop = True
    for i in range(len(window)-1):
        if window[i] > window[i+1]:
            stop = False
    return stop
    
    

def div_data(dataset, proportion_train=0.8):
    n_items = len(dataset)
    n_train = round(n_items * proportion_train)
    n_val = n_items - n_train
    train_data, val_data = random_split(dataset, [n_train, n_val])
    return train_data, val_data


def train_plot(title, train_list, val_list, save_dir):
    if not Path(save_dir).is_dir():
        os.mkdir(save_dir)
    fig, ax = plt.subplots()
    plt.plot(range(len(train_list)), train_list, color='orange', label='Training '+title)
    plt.plot(range(len(val_list)), val_list, color='blue', label='Validation '+title)
    plt.legend()
    plt.xlabel("Number of epochs")
    plt.title(title)
    plt.savefig(os.path.join(save_dir,title.lower()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train emotional speech classifier.')
    parser.add_argument("inpath", type=str, help="Path to directory containing subdirectories 'italian', 'urdu' and 'estonian', each with files 'train_data.csv' and 'test_data.csv' as well as subdirectory 'sgrams' containing input tensors.")
    parser.add_argument("-d", "--device", dest='device', default='cuda:0', help="GPU on which to perform computations.")
    args = parser.parse_args()

    converter = LabelIndexer('AHSN')
    if not Path('./plots').is_dir():
        os.mkdir('plots')

    model_options = {
        'in_channels': 1,
        'n_classes': len(converter),
        'cnn_channels': 64,
        'rnn_hidden': 256,
        'avg_pool_out': 128,  
        'fc1_out': 256,
        'kernel_size': 5,
        'pool_size': 2,
        'conv_stride': 1
    }

    print('Collecting data...\n')

    data_dir = args.inpath  # e.g. '/srv/data/gussodato/emotions/'

    italian_dir = os.path.join(data_dir, 'italian')
    urdu_dir = os.path.join(data_dir, 'urdu')
    estonian_dir = os.path.join(data_dir, 'estonian')

    ita_train = EmoDataset(os.path.join(italian_dir, 'train_data.csv'), os.path.join(italian_dir, 'sgrams/'), converter)
    ita_test =  EmoDataset(os.path.join(italian_dir, 'test_data.csv'), os.path.join(italian_dir, 'sgrams/'), converter)
    urd_train = EmoDataset(os.path.join(urdu_dir, 'train_data.csv'), os.path.join(urdu_dir, 'sgrams/'), converter)
    urd_test = EmoDataset(os.path.join(urdu_dir, 'test_data.csv'), os.path.join(urdu_dir, 'sgrams/'), converter)
    est_train = EmoDataset(os.path.join(estonian_dir, 'train_data.csv'), os.path.join(estonian_dir, 'sgrams/'), converter)
    est_test = EmoDataset(os.path.join(estonian_dir, 'test_data.csv'), os.path.join(estonian_dir, 'sgrams/'), converter)

    ita_model = EmoClassModel(model_options)
    urd_model = EmoClassModel(model_options)
    est_model = EmoClassModel(model_options)

    print('\n\nSize of datasets:')
    print(f'Italian: {len(ita_train)+len(ita_test)} of which {len(ita_train)} for training.')
    print(f'Urdu: {len(urd_train)+len(urd_test)} of which {len(urd_train)} for training.')
    print(f'Estonian: {len(est_train)+len(est_test)} of which {len(est_train)} for training.')

    print('\nLabel distributions:')
    for language, dataset in [('Italian data', ita_train), ('Urdu data', urd_train), ('Estonian data', est_train)]:
        print(f'In {language}:')
        for label in dataset.classnames:
            print(f'{label}: {dataset.proportions[label]}')

    
    # Experiment settings:
    
    italian_options = {
        'experiment_name': 'italian', 
        'model': ita_model,
        'train_data': ita_train,
        'val_data': ita_test,
        'device': torch.device(args.device),
        'epochs': 400,
        'batch_size': 8,
        'lr': 0.001,
        'early_stopping': True,
        'min_epochs': 10,
        'patience': 5,
        'checkpoint_dir': './checkpoints/',
        'plot_dir': './plots/italian'
    }

    urdu_options = {
        'experiment_name': 'urdu',  
        'model': urd_model,
        'train_data': urd_train,
        'val_data': urd_test,
        'device': torch.device(args.device),
        'epochs': 400,
        'batch_size': 8,
        'lr': 0.001,
        'early_stopping': True,
        'min_epochs': 10,
        'patience': 5,
        'checkpoint_dir': './checkpoints/',
        'plot_dir': './plots/urdu'
    }

    estonian_options = {
        'experiment_name': 'estonian',  
        'model': est_model,
        'train_data': est_train,
        'val_data': est_test,
        'device': torch.device(args.device),
        'epochs': 400,
        'batch_size': 8,
        'lr': 0.001,
        'early_stopping': True,
        'min_epochs': 10,
        'patience': 5,
        'checkpoint_dir': './checkpoints/',
        'plot_dir': './plots/estonian'
    }

    print('\n\nTraining models...')
    for experiment in [italian_options, urdu_options, estonian_options]:
        train_losses, eval_losses, train_acc, eval_acc = train_emotion_model(experiment)
        train_plot('Loss', train_losses, eval_losses, experiment['plot_dir'])
        train_plot('Accuracy', train_acc, eval_acc, experiment['plot_dir'])











