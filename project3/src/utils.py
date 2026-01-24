import numpy as np
import os
import json
import matplotlib.pyplot as plt
import pickle
import os

def split_data(features_dict, n_train=5):
    """
    split into train set and test set
    """
    train_data = {}
    test_data = {}
    
    for digit, features_list in features_dict.items():
        # shuffle
        indices = np.random.permutation(len(features_list))
        
        # split
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]
        
        train_data[digit] = [features_list[i] for i in train_indices]
        test_data[digit] = [features_list[i] for i in test_indices]
    
    return train_data, test_data

def plot_confusion_matrix(confusion_matrix, title="Confusion Matrix"):
    """
    plot confusion matrix
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, [str(i) for i in range(10)])
    plt.yticks(tick_marks, [str(i) for i in range(10)])
    
    plt.ylabel('True digit')
    plt.xlabel('Predicted digit')
    
    thresh = confusion_matrix.max() / 2.
    for i in range(10):
        for j in range(10):
            plt.text(j, i, format(confusion_matrix[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if confusion_matrix[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.show()

def save_results(results, filename="results.json"):
    """
    save results to json
    """
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

def load_results(filename="results.json"):
    """
    load data from json
    """
    with open(filename, 'r') as f:
        return json.load(f)


def save_hmm_models(models, path):
    """
    models: dict, e.g. { '0': GaussianHMM, '1': GaussianHMM, ... }
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(models, f)


def load_hmm_models(path):
    with open(path, "rb") as f:
        return pickle.load(f)