import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from models import EmotionCausePairClassifierModel
from sklearn.model_selection import KFold, train_test_split

class Wrapper():
    def __init__(self, args, dataset):
        self.args = args
        self.dataset = dataset
        self.X = dataset.text_embeddings
        self.y = dataset.causes
        self.model = EmotionCausePairClassifierModel(args)

    def run(self):
        # Split the input data (text_embeddings) and output labels (causes)
        X = dataset.text_embeddings
        y = dataset.causes
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.seed)
        k = args.kfolds
        # KFold cross validator
        kf = KFold(n_splits=k, shuffle=True, random_state=self.seed)


