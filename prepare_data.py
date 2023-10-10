import os
import sys
import torch
from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader
import json
import time
import tarfile
import argparse
import pickle

def print_time():
    print('\n----------{}----------'.format(time.strftime("%Y-%m-%d %X", time.localtime())))

def parse(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--task",type=int,default=1,help="Subtask 1 or 2")
    parser.add_argument("--input_dir",type=str,default="./data",help="Path to input directory")
    parser.add_argument("--text_input_dir",type=str,default="text",help="Path to text input files within input_dir")
    parser.add_argument("--video_input_dir",type=str,default="video",help="Path to video input folders within input_dir")
    parser.add_argument("--text_file",type=str,default="Subtask_2_1_train.json",help="Json file for text input within text_input_dir")
    parser.add_argument("--videos_folder",type=str,default="train", help="Folder to mp4 videos within video_inpupt_dir")
    parser.add_argument("--max_sen_len",type=int,default=64,help="Max number of tokens per sentence")
    parser.add_argument("--max_convo_len",type=int,default=35,help="Max number of utterances per conversation")
    parser.add_argument("--embedding_path",type=str,default="./data/saved/embeddings1.pkl",help="Path to already computed embeddings. Defaults to './data/saved/embeddings1.pkl' for subtask 1")
    parser.add_argument("--labels_path",type=str,default="./data/saved/labels1.pkl",help="Path to already computed emotion labels. Defaults to './data/saved/labels1.pkl' for subtask 1")
    parser.add_argument("--lengths_path",type=str,default="./data/saved/lengths1.pkl",help="Path to already computed lengths of conversations. Defaults to './data/saved/lengths1.pkl' for subtask 1")
    parser.add_argument("--causes_path",type=str,default="./data/saved/causes1.pkl",help="Path to already computed lengths of conversations. Defaults to './data/saved/causes1.pkl' for subtask 1")
    parser.add_argument("--given_emotions_path",type=str,default="./data/saved/given_emotions1.pkl",help="Path to already computed lengths of conversations. Defaults to './data/saved/given_emotions1.pkl' for subtask 1")

    all_args = parser.parse_known_args(args)[0]
    return all_args

class CustomDataset(Dataset):
    def __init__(self, args):
        self.device = args.device
        self.max_sen_len = args.max_sen_len
        self.max_convo_len = args.max_convo_len
        self.text_input_path = os.path.join(os.path.join(args.input_dir, args.text_input_dir),args.text_file)
        self.video_input_path = os.path.join(os.path.join(args.input_dir, args.video_input_dir), args.videos_folder)
        self.text_embeddings, self.emotion_labels, self.convo_lengths, self.cause_labels, self.given_emotions = self.get_text_embeddings(args.embedding_path, args.labels_path, args.lengths_path, args.causes_path, args.given_emotions_path)
        self.emotion_idx = dict(zip(['anger', 'disgust', 'fear', 'sadness', 'neutral','joy','surprise'], range(7)))

    def __len__(self):
        return len(self.text_embeddings)

    def __getitem__(self, idx):
        sample = {
            'embeddings': self.text_embeddings[idx], 
            'given_emotion': self.given_emotions[idx], 
            'cause_label': self.cause_labels[idx],
            'num_utterances': self.convo_lengths[idx],
        }
        return sample

    def get_text_embeddings(self, embedding_path, labels_path, lengths_path, causes_path, given_emotions_path):
        if not os.path.exists(embedding_path):
            print("\nComputing embeddings using file at {}".format(self.text_input_path))
            file = open(self.text_input_path, 'r', encoding='utf-8')
            data = json.load(file)
            # remove 
            data = data[:10]

            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
            model.eval()

            text_embeddings = []
            emo_labels = []
            causes = []
            given_emo_utt = []

            for convo in data:
                input_ids = []
                attention_masks = []
                convo_embeddings = []
                labels = []
                # Store the given target emotion utterance index in conversation, 0 based
                emo_utt_num = int(convo['emotion_utterance_ID'].split('utt')[1])
                given_emo_utt.append(emo_utt_num - 1)
                for utt in convo['conversation']:
                    labels.append(utt['emotion'])
                    text = utt['text']
                    # for subtask 1, add emotion label as special token within utt
                    text = text + ' [' + utt['emotion'] + ']'
                    encoded_dict = tokenizer.encode_plus(
                                text,                      # Sentence to split into tokens
                                add_special_tokens = True, # Add special token '[CLS]' and '[SEP]'
                                max_length = self.max_sen_len,           # Pad & truncate all sentences.
                                pad_to_max_length = True,
                                return_attention_mask = True,   # Construct attention masks.
                                return_tensors = 'pt',     # Return pytorch tensors.
                                truncation=True,
                           )
                    # adding the encoded sentence to the list. 
                    input_ids.append(encoded_dict['input_ids']) 
                    # attention mask (to differentiate padding from non-padding).
                    attention_masks.append(encoded_dict['attention_mask'])

                    # calc outputs
                    outputs = model(encoded_dict['input_ids'], encoded_dict['attention_mask'])

                    bert_hidden_states = outputs.hidden_states
                    # average the second to last layer for each token in a sent
                    token_vecs = bert_hidden_states[-2][0]
                    sent_embedding = torch.mean(token_vecs, dim=0) # each of 768 dim
                    convo_embeddings.append(sent_embedding)

                text_embeddings.append(convo_embeddings)
                emo_labels.append(labels)
                num_utt = len(convo_embeddings)
                cause_vec = torch.zeros(num_utt)
                # For every sample given store whether each utt is a cause: true(1) or false(0) 
                for c in convo['cause_utterances']:
                    cause_vec[int(c) - 1] = 1
                causes.append(cause_vec)

            print("\nShape of each sentence embedding for utterance: {}".format(sent_embedding.shape))
            # pad convo_embeddings with zero torch tensors of 768 dim upto max_convo_len
            text_embeddings, lengths = self.pad_conversation_lists(text_embeddings)
            # Save the embeddings and emotion labels
            with open(embedding_path, 'wb') as f:
                pickle.dump(text_embeddings, f)
            with open(labels_path, 'wb') as f:
                pickle.dump(emo_labels, f)
            with open(lengths_path, 'wb') as f:
                pickle.dump(lengths, f)
            with open(causes_path, 'wb') as f:
                pickle.dump(causes, f)
            with open(given_emotions_path, 'wb') as f:
                pickle.dump(given_emo_utt, f)
            return text_embeddings, emo_labels, lengths, causes, given_emo_utt

        else:
            print("\nLoading pre-computed embeddings")
            with open(embedding_path, "rb") as f:
                text_embeddings = pickle.load(f)
            with open(labels_path, "rb") as f:
                emo_labels = pickle.load(f)
            with open(lengths_path, "rb") as f:
                lengths = pickle.load(f)
            with open(causes_path, "rb") as f:
                causes = pickle.load(f)
            with open(given_emotions_path, "rb") as f:
                given_emo_utt = pickle.load(f)
            return text_embeddings, emo_labels, lengths, causes, given_emo_utt
        #emb = get_text_embeddings('a')

    def pad_conversation_lists(self, text_embeddings):
        """Takes a list of conversation embeddings of varying lengths
        and pads them / trims them to max_convo_len defined
        and also returns the actual lengths of conversations"""
        # determine dimension of each utt tensor, 768
        tensor_dim = text_embeddings[0][0].size(-1)
        padding_tensor = torch.zeros(tensor_dim)
        new_text_embeddings = []
        lengths = []

        for convo_list in text_embeddings:
            length = 0
            num_padding = self.max_convo_len - len(convo_list)
            if num_padding > 0:
                padding = [padding_tensor.clone() for _ in range(num_padding)]
                convo_list.extend(padding)
                length = len(convo_list)
            else:
                convo_list = convo_list[:self.max_convo_len]
                length = self.max_convo_len
            new_text_embeddings.append(convo_list)
            lengths.append(length)
        return new_text_embeddings, lengths


def extract_videos(file_name):
    f = tarfile.open(file_name)
    f.extractall('./data/{}/videos'.format(file_name))
    f.close

def main():
    args = parse(sys.argv[1:])
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    args.device = device
    # Create folder for saving embeddings and other computed quantities
    saved_dir_path = os.path.join((args.input_dir), 'saved')
    if not os.path.exists(saved_dir_path):
        os.makedirs(saved_dir_path)
    task1_dataset = CustomDataset(args)

if __name__ == '__main__':
    main()
