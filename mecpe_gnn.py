# Import utility functions
import sys
sys.path.insert(0, './utils/')
from utils import *

# Import modules
import os
import torch
import argparse
import pickle

from models import EmotionCauseClassifierModel
from prepare_data import CustomDataset

# Command line arguments
def parse(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", action="store_false", help='whether to perform training or not', required=False)
    parser.add_argument("--test", action="store_false", help='whether to perform testing or not', required=False)
    parser.add_argument('--choose_emocate', default='',help="whether to predict emotion category")
    # custom transformer model structure
    parser.add_argument('--input_dim_transformer', default=768, help='input dimension for EmbeddingModifierTransformer')
    parser.add_argument('--hidden_dim_transformer', default=256, help='hidden dimension for EmbeddingModifierTransformer')
    parser.add_argument('--num_heads_transformer', default=4, help='number of heads for for EmbeddingModifierTransformer')
    parser.add_argument('--num_layers_transformer', default=4, help='number of layers for EmbeddingModifierTransformer')
    # paths
    parser.add_argument("--log_dir",type=str, default="./logs",help="path to log directory", required=False)
    parser.add_argument("--input_dir",type=str, default='./data', help='path to input directory', required=False)
    parser.add_argument("--output_dir", type=str, default='./output/', required=False)
    # training args
    parser.add_argument("--batch_size", type=int, default=32, help='number of example per batch', required=False)
    parser.add_argument("--epochs", type=int, default=10, help='Number of epochs', required=False)
    parser.add_argument("--lr", type=float, default=0.005, required=False)
    parser.add_argument("--l2_reg", type=float,default=1e-5,required=False,help="l2 regularization")
    parser.add_argument("--no_cuda", action="store_true", help="sets device to CPU", required=False)
    parser.add_argument("--seed", type=int, default=7, required=False)
    parser.add_argument("--emo",type=float,default=1.,help="loss weight of emotion ext.")
    parser.add_argument("--cause",type=float,default=1.0,help="loss weight for cause ext.")

    # Task
    parser.add_argument("--task",type=int,default=1,help="Subtask 1 or 2")
    # Input directory and file names
    parser.add_argument("--text_input_dir",type=str,default="text",help="Path to text input files within input_dir")
    parser.add_argument("--video_input_dir",type=str,default="video",help="Path to video input folders within input_dir")
    parser.add_argument("--text_file",type=str,default="Subtask_2_1_train.json",help="Json file for text input within text_input_dir")
    parser.add_argument("--videos_folder",type=str,default="train", help="Folder to mp4 videos within video_inpupt_dir")
    # Embedding
    parser.add_argument("--embedding_dim",type=float,default=768,help="dimension for word embedding")
    parser.add_argument("--embedding_dim_pos",type=float,default=50,help="dimensions for position embedding")
    # Input structure
    parser.add_argument("--max_sen_len",type=int,default=64,help="Max number of tokens per sentence")
    parser.add_argument("--max_convo_len",type=int,default=35,help="Max number of utterances per conversation")
    # Paths for saving computed values
    parser.add_argument("--embedding_path",type=str,default="./data/saved/embeddings1.pkl",help="Path to already computed embeddings. Defaults to './data/saved/embeddings1.pkl' for subtask 1")
    parser.add_argument("--labels_path",type=str,default="./data/saved/labels1.pkl",help="Path to already computed emotion labels. Defaults to './data/saved/labels1.pkl' for subtask 1")
    parser.add_argument("--lengths_path",type=str,default="./data/saved/lengths1.pkl",help="Path to already computed lengths of conversations. Defaults to './data/saved/lengths1.pkl' for subtask 1")
    parser.add_argument("--causes_path",type=str,default="./data/saved/causes1.pkl",help="Path to already computed lengths of conversations. Defaults to './data/saved/causes1.pkl' for subtask 1")
    parser.add_argument("--given_emotions_path",type=str,default="./data/saved/given_emotions1.pkl",help="Path to already computed lengths of conversations. Defaults to './data/saved/given_emotions1.pkl' for subtask 1")

    all_args = parser.parse_known_args(args)[0]
    return all_args

def run(args):
    # Create dataset for task 1
    dataset = CustomDataset(args)

    model = EmotionCauseClassifierModel(args.input_dim_transformer, args.hidden_dim_transformer, args.num_heads_transformer, args.num_layers_transformer)

def main():
    args = parse(sys.argv[1:])

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    args.device = device

    saved_embed_path = os.path.join(args.input_dir, 'saved')
    if not os.path.exists(saved_embed_path):
        os.makedirs(saved_embed_path)

    run(args)

if __name__ == '__main__':
    main()

