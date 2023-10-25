# LT2326-Project: Learning with audio data

This project was a way for me to get aquainted with a new type of data for machine learning (audio), as well as more complex model architectures than what I have used previously. It contains three sub-projects: 

### Birds
A binary classification problem where the task is to detect bird sounds in an audio clip (more details [here](https://machine-listening.eecs.qmul.ac.uk/bird-audio-detection-challenge/)). Training results can be seen [here](https://github.com/TomBladsjo/LT2326-Project/tree/main/birds).

### Emotions
A multiclass classification problem: given an audio clip of emotional speech, label it as either A (angry), S (sad), H (happy) or N (neutral). For this sub-project i used three separate datasets in three different languages: [EMOVO](http://voice.fub.it/activities/corpora/emovo/index.html) in Italian, [URDU Dataset](https://github.com/siddiquelatif/URDU-Dataset/tree/master) in Urdu and [Estonian Emotional Speech Corpus](https://metashare.ut.ee/repository/download/4d42d7a8463411e2a6e4005056b40024a19021a316b54b7fb707757d43d1a889/) in Estonian. I trained three models with identical architecture, one on each language, and then tested each model on all three languages (results can be seen [here](https://github.com/TomBladsjo/LT2326-Project/tree/main/emotion/plots/testing)).

### ASR
The third sub-project is a first attempt at training a speech recognition model on the [LibriSpeech Dataset](http://www.openslr.org/12).

## To run the scripts:
### Birds
#### Preprocessing
[preprocessing.py](https://github.com/TomBladsjo/LT2326-Project/blob/main/birds/preprocessing.py) preprocesses audio data and saves as tensors (adapted specifically for the [warblr10k](https://machine-listening.eecs.qmul.ac.uk/bird-audio-detection-challenge/) dataset).

Takes command line arguments 
- "inpath": Path to the audio file directory.
- "outpath": Path to directory to save the output sgram files.
- "metadata": Metadata file (csv).

Optional arguments:
- "--column_id", "-c": The ID of the metadata column containing the file names (default "itemid").
- "--duration", "-d": The duration of the output audioclips in milliseconds (default 10000).
- "--sample_rate", "-sr": The sample rate for resampling the audio (default 44100).
#### Training
[birdsounds.py](https://github.com/TomBladsjo/LT2326-Project/blob/main/birds/birdsounds.py) defines and trains a binary audio classification model. 

Takes command line arguments
- "indata_dir": Path to the input sgram files.
- "metadata_file": Metadata file (csv) containing input file IDs and gold labels.
- "checkpoints_dir": Path to directory where model state dicts will be saved.
- "device": The device where training is done.

### Emotions
#### Preprocessing
[preprocess_emotions.py](https://github.com/TomBladsjo/LT2326-Project/blob/main/emotion/preprocess_emotions.py) preprocesses and optionally augments speech data for emotion classification. Adapted to work specifically with [EMOVO](http://voice.fub.it/activities/corpora/emovo/index.html), [URDU Dataset](https://github.com/siddiquelatif/URDU-Dataset/tree/master) and [Estonian Emotional Speech Corpus](https://metashare.ut.ee/repository/download/4d42d7a8463411e2a6e4005056b40024a19021a316b54b7fb707757d43d1a889/). 

Takes command line arguments
- "inpath": Path to directory containing the data. Assumes a structure where "inpath" contains subdirectories 'italian', 'urdu' and 'estonian', each with subdirectory 'data' containing the (unzipped) downloaded data in the structure it came.

Optional:
- "-a", "--augment": If True, also augments the data (default False).

#### Training
[emotion_models.py](https://github.com/TomBladsjo/LT2326-Project/blob/main/emotion/emotion_models.py) trains three models, one on each of the emotional speech datasets.

Takes command line arguments
- "inpath": Path to directory containing train- and test/validation data. Assumes a structure where "inpath" contains subdirectories 'italian', 'urdu' and 'estonian', each with files 'train_data.csv' and 'test_data.csv' as well as subdirectory 'sgrams' containing input tensors.")

Optional:
- "-d", "--device": GPU on which to perform computations (default "cuda:0").

#### Testing
[test_emotions.py](https://github.com/TomBladsjo/LT2326-Project/blob/main/emotion/test_emotions.py) tests the three models on each of the datasets. Plots are saved in subdirectory "/plots/testing/".

Takes command line arguments
- "inpath": Path to directory containing subdirectories 'italian', 'urdu' and 'estonian', each containing file 'test_data.csv' as well as subdirectory 'sgrams' containing input tensors.

Optional:
- "-d", "--device": GPU on which to perform computations (default "cuda:0).

### ASR
#### Preprocessing
[preprocessing_asr.py](https://github.com/TomBladsjo/LT2326-Project/blob/main/asr/preprocessing_asr.py) preprocesses asr training data and saves the audiofiles as tensors. Adapted to work with, and assumes the directory structure of the [LibriSpeech Corpus](http://www.openslr.org/12). If multiple datasets are downloaded (say, train, dev and test set), this script has to be called separately for each dataset. For this project, I used the smaller 100 hour train set and the "clean" dev set for training.

Takes command line arguments
- "inpath": Path to the audio files.
- "outpath": Path to directory to save the output .pt files in.
- "transcript_path": Path to directory to save the transcript file in.

Optional arguments:
- "--sample_rate", "-sr": The sample rate for resampling the audio (default 44100).
- "--alphabet", "-a": Wether to save the set of characters occurring in transcripts (default False).
- "--vocab", "-v": Wether to save the set of words occurring in transcripts (default False).

#### Training
[asr.py](https://github.com/TomBladsjo/LT2326-Project/blob/main/asr/asr.py) trains an ASR model on preprocessed data.

Takes command line arguments
- "traindata": Path to the training input files.
- "train_transcript": File (csv) containing training input file IDs and transcriptions.
- "devdata": Path to the development set input files.
- "dev_transcript": File (csv) containing devset input file IDs and transcriptions.

Optional arguments:
- "-a", "--alphabet": File (txt) containing the permitted characters (default None).
- "-d", "--device": GPU on which to perform computations (default "cuda:0").

#### Testing
There is currently no inference/testing script for the ASR part of the project. This is because the model did not learn well enough during training for it to make sense to attempt any inference (the model would most likely only predict blanks). If I manage to fix that, I might add an inference script here at some later point, but it will not happen within the time frame of this project.





















