import os
import sys
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
from sklearn.model_selection import train_test_split
import csv
import emotion_models



def load_model(resume_file, model):
    if os.path.isfile(resume_file):
        print('Loading model %s'%resume_file)
        checkpoint = torch.load(resume_file)
        model.load_state_dict(checkpoint)
        return model
    else:
        sys.exit(f"No checkpoint found at '{resume_file}'.\nExiting.")
        


def test_model(model, device, testsets):
    """
    Args:
        model: A trained classification model.
        device: A torch.device()
        testsets: A dictionary of test datasets with titles as keys and pytorch Datasets as values.
    Returns:
        A dictionary with dataset titles as keys and inner dictionaries with metric as keys and
        score as value.
    """
    scores = {
        'testset': [],
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1': []
    }
    model.to(device)
    model.eval()
    with torch.no_grad():
        for t in testsets:
            scores['testset'].append(t)
            ts = testsets[t]
            dataloader = DataLoader(ts, batch_size=4, shuffle=False, collate_fn=emotion_models.EmoCollator())
            print(f'Testing model on {t}...')
            preds = []
            truths = []
            for i, batch in enumerate(tqdm(dataloader)):
                try:
                    x, y_true = batch['input'], batch['label']
                    y_pred = model(x.to(device))
                    batch_preds = [torch.argmax(prediction) for prediction in y_pred.cpu()]
                    assert len(y_true) == len(batch_preds)
                    preds.extend(batch_preds)
                    truths.extend(y_true)
                except:
                    print('Something went wrong here, skipping batch!')  # Again, bc somehting is up with the Estonian data.
            scores['Accuracy'].append(round(accuracy_score(truths, preds), 2))
            scores['Precision'].append(round(precision_score(truths, preds, average='weighted'), 2))
            scores['Recall'].append(round(recall_score(truths, preds, average='weighted'), 2))
            scores['F1'].append(round(f1_score(truths, preds, average='weighted'), 2)) 
    return scores


def plot_scores_per_model(score_dict, model_name, save_dir):
    # results grouped by language, labels for score type
    testsets = [language+' data' for language in score_dict['testset']]
    x = np.arange(len(testsets))  # the label locations
    width = 0.23  # the width of the bars
    multiplier = 0
    fig, ax = plt.subplots(layout='constrained')
    
    for key, scores in score_dict.items():
        if key == 'testset':
            continue
        offset = width * multiplier
        rects = ax.bar(x + offset, scores, width, label=key)
        ax.bar_label(rects, padding=3)
        multiplier += 1
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_title(model_name)
    ax.set_xticks(x + width, testsets)
    ax.legend()
    ax.set_ylim(0, 1)

    plt.savefig(os.path.join(save_dir,model_name.lower().replace(' ', '_')))


def plot_score_all_models(score_dict_list, model_name_list, save_dir, score='F1'):
    # results grouped by model, labels for testset
    models = model_name_list
    testsets = score_dict_list[0]['testset']
    scores = [score_dict[score] for score_dict in score_dict_list]
    plot_dict = {testsets[i]: [score[i] for score in scores] for i in range(len(testsets))}

    x = np.arange(len(models))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0
    fig, ax = plt.subplots(layout='constrained')

    for key, scores in plot_dict.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, scores, width, label=key+' data')
        ax.bar_label(rects, padding=3)
        multiplier += 1
    
    ax.set_title('Results, all models')
    ax.set_xticks(x + width, models)
    ax.legend()
    ax.set_ylim(0, 1)

    plt.savefig(os.path.join(save_dir, 'all_models_'+score.lower()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test emotional speech classifiers.')
    parser.add_argument("inpath", type=str, help="Path to directory containing subdirectories 'italian', 'urdu' and 'estonian', each with files 'train_data.csv' and 'test_data.csv' as well as subdirectory 'sgrams' containing input tensors.")
    parser.add_argument("-d", "--device", dest='device', default='cuda:0', help="GPU on which to perform computations.")
    args = parser.parse_args()

    device = torch.device(args.device)
    plot_dir = './plots/testing/'

    basedir = args.inpath  # e.g. '/srv/data/gussodato/emotions/'
    
    print('Collecting data...\n')
    converter = emotion_models.LabelIndexer('AHSN')
    italian_data = emotion_models.EmoDataset(os.path.join(basedir, 'italian/test_data.csv'), os.path.join(basedir, 'italian/sgrams/'), converter)
    urdu_data = emotion_models.EmoDataset(os.path.join(basedir, 'urdu/test_data.csv'), os.path.join(basedir, 'urdu/sgrams/'), converter)
    estonian_data = emotion_models.EmoDataset(os.path.join(basedir, 'estonian/test_data.csv'), os.path.join(basedir, 'estonian/sgrams/'), converter)

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

    
    datasets = {
        'Italian': italian_data,
        'Urdu': urdu_data,
        'Estonian': estonian_data
    }

    print('Testing models...')
    
    scoredicts = []
    model_names = []
    for language in ['Italian', 'Urdu', 'Estonian']:
        model_name = language+' model'
        resume_file = os.path.join('./checkpoints', language.lower()+'_best_model.pt')
        model = emotion_models.EmoClassModel(model_options)
        trained_model = load_model(resume_file, model)

        scores = test_model(trained_model, device, datasets)
        plot_scores_per_model(scores, model_name, plot_dir)
        scoredicts.append(scores)
        model_names.append(model_name)

    plot_score_all_models(scoredicts, model_names, plot_dir, score='F1')
    plot_score_all_models(scoredicts, model_names, plot_dir, score='Accuracy')

    print('Done!')





























