import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from models import EmotionCausePairClassifierModel
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader
import importlib
import random

import prepare_data
importlib.reload(prepare_data)
from prepare_data import CustomDataset
from models import EmotionCausePairClassifierModel

class Wrapper():
    def __init__(self, args, dataset):
        self.dataset = dataset
        self.X = dataset.text_embeddings
        self.y = dataset.cause_labels
        self.true_y_lengths = dataset.lengths
        self.given_emotion_idxs = dataset.given_emotion_idxs
        self.k = args.kfold
        self.seed = args.seed
        self.lr = args.lr
        self.num_epochs = args.num_epochs
        self.l2_reg = args.l2_reg
        self.device = args.device
        self.max_convo_len = args.max_convo_len
        self.batch_size = args.batch_size
        self.threshold = args.threshold
        self.adj = dataset.adj

    def run(self, args):
        # Split the input data (text_embeddings) and output labels (causes) and true lengths
        X_trainval, X_test, y_trainval, y_test, true_y_lengths_trainval, true_y_lengths_test, given_emotion_idxs_trainval, given_emotion_idxs_test = train_test_split(self.X, self.y, self.true_y_lengths, self.given_emotion_idxs, test_size=0.2, random_state=self.seed)
        # KFold cross validator
        kf = KFold(n_splits=self.k, shuffle=True, random_state=self.seed)

        # Prepare train and val datasets for each of the k splits
        train_datasets = []
        val_datasets = []
        for train_idxs, val_idxs in kf.split(X_trainval):
            if torch.is_tensor(train_idxs):
                train_idxs = train_idxs.tolist()
            if torch.is_tensor(val_idxs):
                val_idxs = val_idxs.tolist()

            train_dataset = CustomDataset(
                X_trainval[train_idxs],
                y_trainval[train_idxs],
                true_y_lengths_trainval[train_idxs],
                given_emotion_idxs_trainval[train_idxs],
            )
            val_dataset = CustomDataset(
                X_trainval[val_idxs],
                y_trainval[val_idxs],
                true_y_lengths_trainval[val_idxs],
                given_emotion_idxs_trainval[val_idxs],
            )
            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)

        mean_accuracy_list = []
        mean_precision_list = []
        mean_recall_list = []
        mean_f1_list = []
        best_f1_list = []

        # Iterate through each of the k folds for training and evaluation
        for fold in range(self.k):
            print("\n\n>>>>>>>>>>>>>>>>>>FOLD %d<<<<<<<<<<<<<<<<<<<<<<<<" % (fold + 1))
            self.train_loader = DataLoader(train_datasets[fold], batch_size=self.batch_size, shuffle=True)
            self.val_loader = DataLoader(val_datasets[fold], batch_size=self.batch_size)

            train_losses = []
            val_losses = []
            train_accuracy_list = []
            val_accuracy_list = []
            val_precision_list = []
            val_recall_list = []
            val_f1_list = []

            # Model, loss fn and optimizer
            self.model = EmotionCausePairClassifierModel(args)
            self.model.to(args.device)
            self.criterion = nn.BCELoss(reduction='none') # apply reduction = 'none'?
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

            # Store best f1 across all folds
            best_val_f1 = None

            # Training and Validation loop
            for epoch in range(self.num_epochs):
                # Training
                training_epoch_loss, total_correct, total_samples = self.train(epoch)
                train_accuracy = total_correct / total_samples

                print(f"Epoch {epoch + 1}/{self.num_epochs}, Average Train Loss: {training_epoch_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
                train_losses.append(training_epoch_loss)
                train_accuracy_list.append(train_accuracy)

                # Evaluation
                val_epoch_loss, tp, fp, fn = self.evaluate(epoch)
                if (tp + fp + fn) == 0:
                    val_accuracy = 0.0
                else:
                    val_accuracy = (tp) / (tp + fp + fn)
                if (tp + fp) == 0:
                    val_precision = 0.0
                else:
                    val_precision = (tp)/(tp + fp)
                if (tp + fn) == 0:
                    val_recall = 0.0
                else:
                    val_recall = tp / (tp + fn)
                if (val_precision + val_recall) == 0:
                    val_f1 = 0
                else:
                    val_f1 = (2 * val_precision * val_recall) / (val_precision + val_recall)

                print(f"Epoch {epoch + 1}/{self.num_epochs}, Avg Val Loss: {val_epoch_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")
                val_losses.append(val_epoch_loss)
                val_accuracy_list.append(val_accuracy)
                val_precision_list.append(val_precision)
                val_recall_list.append(val_recall)
                val_f1_list.append(val_f1)

                # Store best f1 across all folds
                if best_val_f1 == None or val_f1 > best_val_f1:
                    best_val_f1 = val_f1

                print("\n>>>>>>>>>>>>>>EPOCH END<<<<<<<<<<<<<<<<")

            # Calculate mean of the validation metrics for this fold
            mean_accuracy = np.mean(val_accuracy_list)
            mean_precision = np.mean(val_precision_list)
            mean_recall = np.mean(val_recall_list)
            mean_f1 = np.mean(val_f1_list)

            print(f"\nFold %d" % (fold + 1))
            print(f"Mean Accuracy: {mean_accuracy:.4f}")
            print(f"Mean Precision: {mean_precision:.4f}")
            print(f"Mean Recall: {mean_recall:.4f}")
            print(f"Mean F1: {mean_f1:.4f}")
            print("\n>>>>>>>>>>>>>>>>>>FOLD END<<<<<<<<<<<<<<<<<<<<<<<<")

            mean_accuracy_list.append(mean_accuracy)
            mean_precision_list.append(mean_precision)
            mean_recall_list.append(mean_recall)
            mean_f1_list.append(mean_f1)
            best_f1_list.append(best_val_f1)

        return mean_accuracy_list, mean_precision_list, mean_recall_list, mean_f1_list, best_f1_list

    def train(self, epoch):
        train_epoch_loss = 0.
        total_correct = 0.
        total_samples = 0.
        with tqdm(total=len(self.train_loader)) as prog_bar:
            for step, data in enumerate(self.train_loader, 0):#step=batch_idx, data=batch
                inputs, labels, true_lengths, given_emotion_ids = data
                inputs, labels, true_lengths, given_emotion_ids = inputs.to(self.device), labels.to(self.device), true_lengths.to(self.device), given_emotion_ids.to(self.device)
                batch_loss, correct = self.update(inputs, labels, true_lengths, given_emotion_ids)
                total_samples += len(inputs)
                total_correct += correct
                train_epoch_loss += batch_loss
                prog_bar.set_description("Epoch: %d\tStep: %d\tTraining Loss: %0.4f" % (epoch+1, step, batch_loss))
                prog_bar.update()
        return train_epoch_loss / len(self.train_loader), total_correct, total_samples

    def update(self, inputs, y_labels, lengths, emotion_ids):
        self.model.train()
        y_preds = self.model(inputs, emotion_ids, self.adj)
        # random_idx = random.randint(0, inputs.shape[0]-1)
        # print("output = ")
        # print(y_preds[random_idx][:lengths[random_idx]])

        y_pred_mask = [torch.ones(length) for length in lengths]
        zero_lens = [(self.max_convo_len - length) for length in lengths]
        y_pred_mask = torch.stack([t for t in [torch.cat((t1, torch.zeros(l)),dim=0) for t1, l in zip(y_pred_mask, zero_lens)]])
        loss = self.criterion(y_preds, y_labels)
        masked_loss = torch.sum(loss * y_pred_mask)
        loss = masked_loss / y_pred_mask.sum() #only consider the outputs for actual utt emb, not paddings
        binary_y_preds = (y_preds > self.threshold).float()
        correct = (y_preds == y_labels).sum().item()
        # print(f"Correct = {correct}")

        self.model.zero_grad()
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()
        return loss.item(), correct

    def evaluate(self, epoch):
        val_epoch_loss = 0.
        total_samples = 0
        tot_tp = 0
        tot_fp = 0
        tot_fn = 0
        with tqdm(total=len(self.val_loader)) as prog_bar:
            with torch.no_grad():
                for step, data in enumerate(self.val_loader, 0):#step=batch_idx, data=batch
                    inputs, labels, true_lengths, given_emotion_ids = data
                    inputs, labels, true_lengths, given_emotion_ids = inputs.to(self.device), labels.to(self.device), true_lengths.to(self.device), given_emotion_ids.to(self.device)
                    batch_loss, tp, fp, fn = self.update_val(inputs, labels, true_lengths, given_emotion_ids)
                    total_samples += len(inputs)
                    tot_tp += tp
                    tot_fp += fp
                    tot_fn += fn
                    val_epoch_loss += batch_loss
                    prog_bar.set_description("Epoch: %d\tStep: %d\tValidation Loss: %0.4f" % (epoch+1, step, batch_loss))
                    prog_bar.update()
        return val_epoch_loss / len(self.val_loader), tot_tp, tot_fp, tot_fn

    def update_val(self, inputs, y_labels, lengths, emotion_ids):
        self.model.eval()
        y_preds = self.model(inputs, emotion_ids, self.adj)
        y_pred_mask = [torch.ones(length) for length in lengths]
        zero_lens = [(self.max_convo_len - length) for length in lengths]
        y_pred_mask = torch.stack([t for t in [torch.cat((t1, torch.zeros(l)),dim=0) for t1, l in zip(y_pred_mask, zero_lens)]])
        loss = self.criterion(y_preds, y_labels)
        masked_loss = torch.sum(loss * y_pred_mask)
        loss = masked_loss / y_pred_mask.sum() #only consider the outputs for actual utt emb, not paddings
        binary_y_preds = (y_preds > self.threshold).float()
        tp, fp, fn = self.tp_fp_fn(binary_y_preds, y_labels, y_pred_mask)
        return loss.item(), tp, fp, fn

    def tp_fp_fn(self, predictions, labels, mask):
        mask = mask.to(torch.int)
        predictions_flat = predictions[mask].to(torch.int)
        labels_flat = labels[mask].to(torch.int)

        tp = torch.sum(predictions_flat & labels_flat)
        fp = torch.sum(predictions_flat & ~labels_flat)
        fn = torch.sum(~predictions_flat & labels_flat)

        return tp, fp, fn

    #define functions for saving and loading models per fold

