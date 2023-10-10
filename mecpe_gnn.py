# Import utility functions
import sys
sys.path.insert(0, './utils/')
from utils import *

# Import modules
import os
import torch
import argparse
from models import EmbeddingModifierTransformer
import pickle

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
    # Path for loading precomputed embeddings
    parser.add_argument("--embedding_path",type=str,default="./data/saved/embeddings1.pkl",help="Path to already computed embeddings. Defaults to './data/saved/embeddings1.pkl' for subtask 1")


    all_args = parser.parse_known_args(args)[0]
    return all_args

def main():
    args = parse(sys.argv[1:])

    transformer_model = EmbeddingModifierTransformer(args.input_dim_transformer, args.hidden_dim_transformer, args.num_heads_transformer, args.num_layers_transformer)
    # Load precomputed input embeddings
    with open(args.embedding_path, 'rb') as file:
        input_embeddings= pickle.load(file)
    print("input_embeddings shape : {}".format(input_embeddings.shape))
    modified_embeddings = transformer_model(input_embeddings)
    print("modified_embeddings shape : {}".format(modified_embeddings.shape))

if __name__ == '__main__':
    main()
