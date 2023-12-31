import pandas as pd
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen

DATASET_LINK = 'https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip'

def download_dataset():
    '''Function to download the dataset from the link'''
    resp = urlopen(DATASET_LINK)
    zip_file = ZipFile(BytesIO(resp.read()))
    zip_file.extractall('data/raw')

def make_intermediate_dataset():
    '''Function to make the intermediate dataset from the raw dataset'''
    # reading the data from the tsv file
    df = pd.read_csv('data/raw/filtered.tsv', sep='\t')
    
    # taking only rows with high similarity and where reference toxicity is higher than translation toxicity 
    sents = df[(df['similarity'] > 0.8) & (df['ref_tox'] - df['trn_tox'] > 0.2) & (df['ref_tox'] > 0.7)]
    sents = sents[['reference', 'translation']]
    sents.columns = ['reference', 'translation']

    # saving the data to intermediate folder
    sents.to_csv('data/interim/filtered.csv', index=False)

def train_test_split():
    '''Function to split the data into train and test sets'''
    # reading the data from the csv file
    df = pd.read_csv('data/interim/filtered.csv')

    # splitting the data into train and test sets
    train = df.sample(frac=0.95, random_state=1337)
    test = df.drop(train.index)

    # saving the train and test sets
    train.to_csv('data/interim/train.csv', index=False)
    test.to_csv('data/interim/test.csv', index=False)

def make_toxic_words_dataset():
    '''Function to make the toxic words dataset'''
    # reading the data from the txt files
    negative_words = open('data/external/negative_words.txt').read().split('\n')
    toxic_words = open('data/external/toxic_words.txt').read().split('\n')
    toxic_words.extend(negative_words)

    # cleaning the toxic words
    toxic_words = [w for w in toxic_words if w.isalnum() and len(w) > 1]
    toxic_words = list(set(toxic_words))

    # saving the toxic words to a file
    with open('data/interim/toxic_words.txt', 'w') as f:
        to_file = '\n'.join(toxic_words)
        f.write(to_file)

def make_positive_words_dataset():
    '''Function to make the positive words dataset'''
    # reading the data from the txt files
    positive_words = open('data/external/positive_words.txt').read().split('\n')

    # cleaning the data
    positive_words = [w for w in positive_words if w.isalnum() and len(w) > 1]
    positive_words = list(set(positive_words))

    # saving the positive words to a file
    with open('data/interim/positive_words.txt', 'w') as f:
        to_file = '\n'.join(positive_words)
        f.write(to_file)

if __name__ == '__main__':
    print('Downloading the dataset...')
    download_dataset()
    make_intermediate_dataset()
    train_test_split()
    make_toxic_words_dataset()
    make_positive_words_dataset()
    print('Done!')