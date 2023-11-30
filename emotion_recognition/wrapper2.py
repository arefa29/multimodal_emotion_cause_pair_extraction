import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from data_loader2 import *
import importlib
import random
import wandb
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report

from models import EmotionRecognitionModel
from transformers import get_linear_schedule_with_warmup

class Wrapper():
    def __init__(self, args):
        self.k = args.kfold
        self.seed = args.seed
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.num_epochs = args.num_epochs
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.l2_reg = args.l2_reg
        self.device = args.device
        self.max_convo_len = args.max_convo_len
        self.batch_size = args.batch_size
        self.warmup_proportion = args.warmup_proportion
        self.emotion_labels = ['anger', 'disgust', 'fear', 'sadness', 'neutral','joy','surprise']
        self.emotion_idx = dict(zip(['anger', 'disgust', 'fear', 'sadness', 'neutral','joy','surprise'], range(7)))

    def run(self, args):
        emotion_aprfb = {'acc': [], 'p': [], 'r': [], 'f': [], 'b': []}
        for fold_id in range(1, self.k + 1):
            wandb.init(
                project="mecpe_emotion_recog",
                config={
                "epochs":args.num_epochs,
                "lr":args.lr,
                "batch_size":args.batch_size,
                "threshold_emo":args.threshold_emo,
                "threshold":args.threshold,
                "max_convo_len":args.max_convo_len,
                },
                entity='arefa2001',
                name=f"fold{fold_id}",
                reinit=True,
            )
            print("\n\n>>>>>>>>>>>>>>>>>>FOLD %d<<<<<<<<<<<<<<<<<<<<<<" % (fold_id))
            self.train_loader = build_train_data(args, fold_id)
            self.val_loader = build_inference_data(args, fold_id, data_type='valid')
            self.test_loader = build_inference_data(args, fold_id, data_type='test')

            train_losses = []
            val_losses = []
            train_accuracy_list = []
            val_accuracy_list = []
            val_precision_list = []
            val_recall_list = []
            val_f1_list = []
            # Store best f1 across all epochs
            best_val_f1_e = None

            # Model, loss fn and optimizer
            self.ohe = OneHotEncoder(categories=self.emotion_labels)
            self.model = EmotionRecognitionModel(args)
            self.model.to(self.device)
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            self.num_update_steps = len(self.train_loader) // self.gradient_accumulation_steps * self.num_epochs
            self.warmup_steps = self.warmup_proportion * self.num_update_steps
            scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.num_update_steps)
            self.model.zero_grad()

            # Training and Validation loop
            for epoch in range(self.num_epochs):
                training_epoch_loss, total_correct, total_samples = self.train(epoch)
                train_accuracy_e = total_correct / total_samples

                print(f"Epoch {epoch + 1}/{self.num_epochs}, Average Train Loss: {training_epoch_loss:.4f}, Train Accuracy Emotion: {train_accuracy_e:.4f}")
                train_losses.append(training_epoch_loss)
                train_accuracy_list.append(train_accuracy_e)

                # Evaluation
                val_epoch_loss, preds_e, true_e = self.evaluate(epoch)
                print(f"Epoch {epoch + 1}/{self.num_epochs}, Avg Val Loss: {val_epoch_loss:.4f}\n")

                print("\n>>>>>>>>>>>>>EPOCH END<<<<<<<<<<<<<<<<")

            wandb.finish()

        return emotion_aprfb

    def train(self, epoch):
        train_epoch_loss = 0.
        total_correct = 0.
        total_samples = 0.
        with tqdm(total=len(self.train_loader)) as prog_bar:
            for step, batch in enumerate(self.train_loader, 1):#step=batch_idx, data=batch
                adj_b, convo_len_b, y_emotions_b, y_mask_b, \
                    bert_token_b, bert_segment_b, bert_masks_b, bert_utt_b = batch

                batch_loss, correct_e, y_emotions_b_masked = self.update(step, bert_token_b, bert_segment_b, bert_masks_b, bert_utt_b, convo_len_b, adj_b, y_emotions_b, y_mask_b)
                samples = len(y_emotions_b_masked)
                total_samples += samples
                total_correct += correct_e
                train_epoch_loss += batch_loss
                prog_bar.set_description("Epoch: %d\tStep: %d\tTraining Loss: %0.4f" % (epoch+1, step, batch_loss))

                wandb.log({"epoch":epoch+1, "step_train_loss":batch_loss, "step_train_acc_e":correct_e/samples})
                prog_bar.update()
        return train_epoch_loss / len(self.train_loader), total_correct, total_samples

    def update(self, step, bert_token_b, bert_segment_b, bert_masks_b, bert_utt_b, convo_len_b, adj_b, y_emotions_b, y_mask_b):
        y_mask_b = torch.tensor(y_mask_b).bool().to(self.device)
        self.model.train()
        preds_e = self.model(bert_token_b, bert_segment_b, bert_masks_b, bert_utt_b, convo_len_b, adj_b, y_mask_b)
        y_emotions_b = torch.tensor(y_emotions_b, dtype=torch.float32).to(self.device)
        y_emotions_b = y_emotions_b.type(torch.LongTensor).to(self.device)
        y_emotions_b = y_emotions_b.masked_select(y_mask_b)
        # print("y emotions b dev")
        # print(y_emotions_b.is_cuda)
        # print("preds")
        # print(preds_e.is_cuda)

        loss_e = self.criterion(preds_e, y_emotions_b)
        loss = loss_e
        loss = loss / self.gradient_accumulation_steps

        correct_e = (torch.argmax(preds_e, 1) == y_emotions_b).float().sum()

        loss.backward()
        if (step + 1) % self.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss.item(), correct_e, y_emotions_b

    def evaluate(self, epoch):
        val_epoch_loss = 0.
        y_preds_e = []
        true_e = []
        with tqdm(total=len(self.val_loader)) as prog_bar:
            with torch.no_grad():
                for step, batch in enumerate(self.val_loader, 1):# step=batch_idx, data=batch
                    adj_b, convo_len_b, y_emotions_b, y_mask_b,\
                    bert_token_b, bert_segment_b, bert_masks_b, bert_utt_b = batch
                    batch_loss, true_e_step, preds_e_step = self.update_val(epoch, bert_token_b, bert_segment_b, bert_masks_b, bert_utt_b, convo_len_b, adj_b, y_emotions_b, y_mask_b)

                    val_epoch_loss += batch_loss
                    y_preds_e.extend(preds_e_step.cpu().numpy())
                    true_e.extend(true_e_step.cpu().numpy())

                    prog_bar.set_description("Epoch: %d\tStep: %d\tValidation Loss: %0.4f" % (epoch+1, step, batch_loss))
                    prog_bar.update()

        return val_epoch_loss / len(self.val_loader), y_preds_e, true_e

    def update_val(self, epoch, bert_token_b, bert_segment_b, bert_masks_b, bert_utt_b, convo_len_b, adj_b, y_emotions_b,y_mask_b):
        y_mask_b = torch.tensor(y_mask_b).bool().to(self.device)
        self.model.eval()
        preds_e = self.model(bert_token_b, bert_segment_b, bert_masks_b, bert_utt_b, convo_len_b, adj_b, y_mask_b)
        y_emotions_b = torch.tensor(y_emotions_b, dtype=torch.float32).to(self.device)
        y_emotions_b = y_emotions_b.type(torch.LongTensor).to(self.device)
        y_emotions_b = y_emotions_b.masked_select(y_mask_b)

        loss_e = self.criterion(preds_e, y_emotions_b)
        loss = loss_e

        preds_e = torch.argmax(preds_e, 1)
        classification_rep = classification_report(y_emotions_b.cpu().numpy(), preds_e.cpu().numpy(), labels=[0,1,2,3,4,5,6], target_names=self.emotion_idx.keys(), output_dict=True)

        wandb.log({"epoch": epoch+1, "step_val_loss":loss.item()})
        for class_name, vals in classification_rep.items():
            if isinstance(vals, dict):
                wandb.log({f'{class_name}_precision':vals['precision'], f'{class_name}_recall':vals['recall'],f'{class_name}_f1':vals['f1-score'], f'{class_name}_support':vals['support']})
            else:
                wandb.log({f'{class_name}': vals})

        return loss.item(), y_emotions_b, preds_e

    def tp_fp_fn(self, preds_e, y_emotions_b_ohe):

        tp = torch.sum(predictions_e & labels_e)
        fp = torch.sum(predictions_e & ~labels_e)
        fn = torch.sum(~predictions_e & labels_e)

        return tp, fp, fn

    # define functions for saving and loading models per fold

    def accuracy(self, tp, fp, fn):
        if (tp + fp + fn) == 0:
            acc = 0.0
        else:
            acc = (tp) / (tp + fp + fn)
        return acc

    def precision(self, tp, fp):
        if (tp + fp) == 0:
            prec = 0.0
        else:
            prec = (tp)/(tp + fp)
        return prec

    def recall(self, tp, fn):
        if (tp + fn) == 0:
            rec = 0.0
        else:
            rec = tp / (tp + fn)
        return rec

    def f1_score(self, precision, recall):
        if(precision + recall) == 0:
            return 0.0
        else:
            return (2 * precision * recall) / (precision + recall)

    def one_hot_encode(self, vec):
        a = np.array(vec, dtype=int)
        b = np.zeros((a.size, a.max() + 1))
        b[np.arange(a.size), a] = 1.

        return b
