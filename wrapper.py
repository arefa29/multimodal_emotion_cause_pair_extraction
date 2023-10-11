import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from models import EmotionCausePairClassifierModel
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader
from prepare_data import CustomDataset

class Wrapper():
    def __init__(self, args, dataset):
        self.dataset = dataset
        self.X = dataset.text_embeddings
        self.y = dataset.cause_labels
        self.true_y_lengths = dataset.lengths
        self.k = args.kfolds
        self.seed = args.seed
        self.lr = args.lr
        self.num_epochs = args.num_epochs
        self.l2_reg = args.l2_reg
        self.device = args.device
        self.max_convo_len = args.max_convo_len

    def run(self, args):
        # Split the input data (text_embeddings) and output labels (causes) and true lengths
        X_trainval, X_test, y_trainval, y_test, true_y_lengths_trainval, true_y_lengths_test = train_test_split(self.X, self.y, self.true_y_lengths, test_size=0.2, random_state=self.seed)
        # KFold cross validator
        kf = KFold(n_splits=self.k, shuffle=True, random_state=self.seed)

        # Prepare train and val datasets for each of the k splits
        train_datasets = []
        val_datasets = []
        for train_idxs, val_idxs in kf.split(X_trainval):
            train_dataset = CustomDataset(
                X_trainval[train_idxs], 
                y_trainval[train_idxs], 
                true_y_lengths_trainval[train_idxs]
            )
            val_dataset = CustomDataset(
                X_trainval[val_idxs], 
                y_trainval[val_idxs], 
                true_y_lengths_trainval[val_idxs]
            )
            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)

        train_losses = []
        val_losses = []
        accuracy_list = []
        precision_list = []
        recall_list = []
        f1_list = []

        # Iterate through each of the k folds for training and evaluation
        for fold in range(self.k):
            print("\n\n>>>>>>>>>>>>>>>>>>FOLD %d<<<<<<<<<<<<<<<<<<<<<<<<" % (fold))
            self.train_loader = DataLoader(train_datasets[fold], batch_size=self.batch_size, shuffle=True)
            self.val_loader = DataLoader(validation_datasets[fold], batch_size=self.batch_size)

            # Model, loss fn and optimizer
            self.model = EmotionCausePairClassifierModel(args)
            self.model.to(args.device)
            self.criterion = nn.BCELoss(reduction='none') # apply reduction = 'none'?
            self.optimizer = optim.Adam(model.parameters(), lr=self.lr)

            # Training and Validation loop
            for epoch in range(self.num_epochs):
                training_epoch_loss = self.train(epoch)

                print(f"Epoch {epoch + 1}/{self.num_epochs}, Average Train Loss: {training_epoch_loss:.4f}")

                # Evaluation
                correct = 0.
                predicted = 0.
                
                print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {average_loss:.4f}, Train Accuracy: {accuracy:.4f}, Train Precision: {precision:.4f}, Train Recall: {recall:.4f}, Train F1: {f1:.4f}")

            print("\n>>>>>>>>>>>>>>>>>>FOLD END<<<<<<<<<<<<<<<<<<<<<<<<")

    def train(self, epoch):
        train_epoch_loss = 0.
        with tqdm(total=len(self.train_loader[0])) as prog_bar:
            for step, data in enumerate(self.train_loader, 0):#step=batch_idx, data=batch
                inputs, labels, true_lengths = data
                inputs, labels, true_lengths = inputs.to(self.device), labels.to(self.device), true_lengths.to(self.device)
                batch_loss = self.update(inputs, labels, true_lengths)
                train_epoch_loss += batch_loss
                prog_bar.set_description("Epoch: %d\tStep: %d\tTraining Loss: %0.4f" % (epoch, step, batch_loss))
                prog_bar.update()
        return train_epoch_loss / len(self.train_loader)

    def update(self, inputs, y_labels, lengths):
        self.model.train()
        y_preds = self.model(inputs)
        y_pred_mask = torch.stack([torch.ones(length) for length in lengths])
        zero_lens = [(self.max_convo_len - length) for length in lengths]
        y_pred_mask = torch.stack([t for t in [torch.cat((t1, torch.zeros(l) for l in zero_lens),dim=0) for t1 in y_pred_mask]])
        loss = self.criterion(y_preds, y_labels)
        masked_loss = torch.sum(loss * y_pred_mask)
        loss = masked_loss / y_pred_mask.sum() #only consider the outputs for actual utt emb

        self.model.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
