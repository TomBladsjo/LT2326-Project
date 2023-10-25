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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
from pathlib import Path


class SoundDataset(Dataset):
    def __init__(self, metadata_csv, indata_dir):
        metadata = pd.read_csv(metadata_csv)
        self.files = metadata['itemid']
        self.labels = metadata['hasbird']
        self.data_dir = indata_dir if indata_dir[-1] == '/' else indata_dir+'/'
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        
        file = self.data_dir + self.files[idx] + '.pt'
        x = torch.load(file)
        y = self.labels[idx]
  
        return x, y


def div_data(dataset, proportion_train):
    n_items = len(dataset)
    n_train = round(n_items * proportion_train)
    n_val = n_items - n_train
    train_data, val_data = random_split(dataset, [n_train, n_val])
    return train_data, val_data



class AudioClassifier(nn.Module):
    def __init__(self, in_channels=1, hidden_1=20, hidden_2=50, kernel_size=5, pool_size=2, output_dim=1, conv_stride=1, pool_stride=None):
        super(AudioClassifier, self).__init__()

        # Layers:

        # 1st conv-relu-pool
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_1,
			kernel_size=kernel_size)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=pool_size, stride=pool_stride)


        # 2nd conv-relu-pool
        self.conv2 = nn.Conv2d(in_channels=hidden_1, out_channels=hidden_2,
			kernel_size=kernel_size)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=pool_size, stride=pool_stride)


        # fc-relu
        self.fc1 = nn.Linear(137800, 500) 
        self.relu3 = nn.ReLU()

        # fc-logsoftmax
        self.fc2 = nn.Linear(500, 1) 
        # self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        #raise embed()
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        
        x = self.fc2(x)
        # x = self.sigmoid(x)
        
        return x.squeeze()




def train_classifier(model, device, train_dataloader, eval_dataloader, checkpoint_path, max_epochs=100, min_epochs=10, lr=0.001, stop_after=3, positive_weight=0.25):
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=positive_weight).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    if not Path(checkpoint_path).is_dir():
        os.mkdir(checkpoint_path)
    print('Training...')
    
    best_epoch = 0
    train_accuracies = []
    val_accuracies = []
    precisions = []
    recalls = []
    f1s = []
    losses = []
    last_epoch = max_epochs
    for epoch in range(max_epochs):
        model.to(device)
        model.train()
        total_loss = 0
        y_trues = []
        y_preds = []
        for i, batch in enumerate(train_dataloader):
            
            x = batch[0].to(device)
            y_true = batch[1]

            y_pred = model(x)

            loss = loss_fn(y_pred, y_true.float().to(device))
            total_loss += loss.item()
            y_binary = [int(pred > 0.5) for pred in y_pred]
            y_trues += list(y_true)
            y_preds += y_binary
     
            # compute gradients
            loss.backward()
            # update parameters
            optimizer.step()
            # reset gradients
            optimizer.zero_grad()

        train_accuracies.append(accuracy_score(y_trues, y_preds))
        losses.append(total_loss/len(train_dataloader.dataset))

        model.eval()

        with torch.no_grad():
            val_acc, prec, rec, f1 = binary_eval(model, device, eval_dataloader)

        # val_accuracies.append(val_acc)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

        print(f'\n-- Epoch {epoch} ---\nAverage loss: {round(losses[-1], 4)}\nTraining set accuracy: {round(train_accuracies[-1],4)}\nValidation set accuracy: {round(val_acc, 4)}\nPrecision: {prec}\nRecall: {rec}\nF1: {f1}\nProportion positive guesses: {sum(y_preds)/len(y_preds)}')
        
        # if new accuracy is better than last, save weights
        if len(val_accuracies)>0 and val_acc > max(val_accuracies):
            torch.save(model.cpu().state_dict(), checkpoint_path+'/best_model_weights.pt')
            best_epoch = epoch
        if len(val_accuracies)>0 and val_acc > val_accuracies[-1]:
            torch.save(model.cpu().state_dict(), checkpoint_path+'/latest_good_model_weights.pt')
        elif epoch >= min_epochs-1:
            if stopping_criterion(val_accuracies, val_acc, stop_after):
                print(f'Early stopping after {stop_after} epochs without accuracy improvment on validation set.')
                last_epoch = epoch+1
                val_accuracies.append(val_acc)
                break
        val_accuracies.append(val_acc)

    print(f'Done training. \nTotal epochs: {last_epoch} \nBest epoch: {best_epoch}')
    return train_accuracies, val_accuracies, losses, last_epoch


def stopping_criterion(acc_list, new_acc, num_decrease):
    """
    Returns True if accuracy has steadily decreased for num_decrease epochs.
    acc_list: A list of previous validation set accuracies.
    new_acc: The latest validation set accuracy.
    num_decrease: The number of epochs for the accuracy to have decreased in order to stop training.
    """
    window = acc_list[-num_decrease:]+[new_acc]
    stop = True
    for i in range(1, len(window)):
        if window[i] > window[i-1]:
            stop = False
    return stop
    


def binary_eval(model, device, dataloader, threshold=0.5):
    model.to(device)
    y_trues = []
    y_preds = []
    for i, batch in enumerate(dataloader):
        y_pred = model(batch[0].to(device))
        y_true = batch[1]
        y_binary = [int(pred > threshold) for pred in y_pred]
        y_trues += list(y_true)
        y_preds += y_binary
    
    return accuracy_score(y_trues, y_preds), precision_score(y_trues, y_preds), recall_score(y_trues, y_preds), f1_score(y_trues, y_preds)






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Binary classification model')
    parser.add_argument("indata_dir", type=str, help="Path to the input sgram files.")
    parser.add_argument("metadata_file", type=str, help="Metadata file (csv) containing input file IDs and gold labels.")
    parser.add_argument("checkpoints_dir", type=str, help="Path to directory where model state dicts will be saved.")
    parser.add_argument("device", type=str, help="The device where training is done.")
    args = parser.parse_args()


    model = AudioClassifier()
    device = torch.device(args.device)
    all_data = SoundDataset(args.metadata_file, args.indata_dir)
    train_data, val_data = div_data(all_data, 0.8)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    n_samples = len(all_data)
    n_positive = all_data.labels.sum()
    ratio = (n_samples-n_positive)/n_positive
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False)

    print(f'The dataset contains {n_samples} samples of which {n_positive} are positive. \nThe ratio of negative to positive is {ratio}.')

    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        train_acc, val_acc, loss, last_epoch = train_classifier(model, device, train_loader, val_loader, args.checkpoints_dir, positive_weight=torch.Tensor([ratio]))
        
        # Plot accuracy:
        fig, ax = plt.subplots()
        ax.set(ylim=(0, 1))
        plt.plot(range(len(train_acc)), train_acc, color='orange', label='Training set')
        plt.plot(range(len(val_acc)), val_acc, color='blue', label='Validation set')
        plt.legend()
        plt.xlabel("Number of epochs")
        plt.title("Accuracy")
        plt.savefig('accuracy_plot')
    
        # Plot loss:
        plt.clf()
        fig, ax = plt.subplots()
        plt.plot(range(len(loss)), loss, color='green', label='Training loss')
        plt.legend()
        plt.xlabel("Number of epochs")
        plt.title("Loss")
        plt.savefig('loss_plot')

    print(prof)




