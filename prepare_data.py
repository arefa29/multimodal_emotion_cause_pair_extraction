import os
import sys
import torch
from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset, Dataloader
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
    parser.add_argument("--input_dir",type=str,default="./data",help="path to input directory")
    parser.add_argument("--text_input_dir",type=str,default="text",help="path to text input files within input_dir")
    parser.add_argument("--video_input_dir",type=str,default="video",help="path to video input folders within input_dir")
    parser.add_argument("--text_file",type=str,default="Subtask_2_1_train.json",help="json file for text input within text_input_dir")
    parser.add_argument("--videos_folder",type=str,default="train", help="folder to mp4 videos within video_inpupt_dir")
    parser.add_argument("--max_sen_len",type=int,default=64,help="max number of tokens per sentence")
    parser.add_argument("--max_convo_len",type=int,default=45,help="max number of utterances per conversation")
    parser.add_argument("--embedding_path",type=str,default="./data/embeddings1.pkl",help="path to already computed embeddings. defaults to './data/embeddings1.pkl' for subtask 1")
    parser.add_argument("--labels_path",type=str,default="./data/labels1.pkl",help="path to already computed emotion labels. defaults to './labels1.pkl' for subtask 1")

    all_args = parser.parse_known_args(args)[0]
    return all_args

class CustomDataset(Dataset):
    def __init__(self, args):
        self.device = args.device
        self.max_sen_len = args.max_sen_len
        self.max_convo_len = args.max_convo_len
        self.text_input_path = os.path.join(os.path.join(args.input_dir, args.text_input_dir),args.text_file)
        self.video_input_path = os.path.join(os.path.join(args.input_dir, args.video_input_dir), args.videos_folder)
        self.text_embeddings, self.emotion_labels = get_text_embeddings(args.embedding_path)
        self.emotion_idx = dict(zip(['neutral', 'anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise'], range(7)))

    def __len__(self):
        return len(self.text_embeddings)

    def __getitem__(self, idx):
        sample = {'data': self.text_embeddings[idx], 'label':self.emotion_idx[self.emotion_labels[idx]]}
        return sample

    def get_text_embeddings(embedding_path):
        if not os.path.exists(embedding_path):
            print("Computing embeddings using file at {}".format(self.text_input_path))
            file = open(self.text_input_path, 'r', encoding='utf-8')
            data = json.load(file)
            # remove 
            data = data[:50]

            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
            model.eval()

            text_embeddings = []
            emotion_labels = []

            for convo in data:
                input_ids = []
                attention_masks = []
                convo_embeddings = []
                labels = []
                for utt in convo['conversation']:
                    labels.append(utt['emotion'])
                    text = utt['text']
                    encoded_dict = tokenizer.encode_plus(
                                text,                      # Sentence to split into tokens
                                add_special_tokens = True, # Add special token '[CLS]' and '[SEP]'
                                max_length = self.max_sen_len,           # Pad & truncate all sentences.
                                pad_to_max_length = True,
                                return_attention_mask = True,   # Construct attention masks.
                                return_tensors = 'pt',     # Return pytorch tensors.
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
                    sent_embedding = torch.mean(token_vecs, dim=0)
                    convo_embeddings.append(sent_embedding)
                text_embeddings.append(convo_embeddings)
                emotion_labels.append(labels)

            print("Number of conversations: {}".format(len(text_embeddings)))
            print("Shape of each embedding for utterance: {}".format(sent_embedding.shape))
            # Save the embeddings
            with open(embedding_path, 'wb') as f:
                pickle.dump(text_embeddings, f)
            with open(labels_path, 'wb') as f:
                pickle.dump(emotion_labels, f)
            return text_embeddings, emotion_labels

        else:
            print("Loading pre-computed embeddings")
            with open(embedding_path, "rb") as f:
                text_embeddings = pickle.load(f)
            return text_embeddings
        #emb = get_text_embeddings('a')

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

if __name__ == '__main__':
    main()
