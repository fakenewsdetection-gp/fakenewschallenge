import matplotlib.pyplot as plt
from csv import DictWriter
import pandas as pd


label_ref = {'agree': 0, 'disagree': 1, 'discuss': 2, 'unrelated': 3}
label_ref_rev = {0: 'agree', 1: 'disagree', 2: 'discuss', 3: 'unrelated'}


def plot_history(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()


def save_predictions(pred, file):
    """
    Save predictions to CSV file.

    Args:
        pred: numpy array, of numeric predictions
        file: str, filename + extension
    """
    with open(file, 'w') as csvfile:
        fieldnames = ['Stance']
        writer = DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for instance in pred:
            writer.writerow({'Stance': label_ref_rev[instance]})


def load_features(name, header='train'):
    filename = '.'.join([header, name, 'pickle'])
    return pd.read_pickle(filename)


def save_features(df, name, header='train'):
    filename = '.'.join([header, name, 'pickle'])
    df.to_pickle(filename)
