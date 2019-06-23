from pylab import *
from csv import DictWriter


label_ref = {'stanced': 0, 'discuss': 1, 'unrelated': 2}
label_ref_rev = {0: 'stanced', 1: 'discuss', 2: 'unrelated'}


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
            writer.writerow({'Stance': instance})
