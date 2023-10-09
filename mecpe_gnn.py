# Import utility functions
import sys
sys.path.insert(0, './utils/')
from utils import *

# Import modules
import os
import torch
import argparse

# Command line arguments
def parse(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", action="store_false", help='whether to perform training or not', required=False)
    parser.add_argument("--test", action="store_false", help='whether to perform testing or not', required=False)
    # Embedding
    parser.add_argument("--embedding_dim",type=float,default=300,help="dimension for word embedding")
    parser.add_argument("--embedding_dim_pos",type=float,default=50,help="dimensions for position embedding")
    # input struct
    parser.add_argument('--max_sen_len', type=float,default=35, help='max number of tokens per sentence')
    parser.add_argument('--max_doc_len',type=float, default= 35, help='max number of sentences per document')
    parser.add_argument('--max_sen_len_bert',type=float,default= 40, help='max_number_of_tokens_per_sentence')
    parser.add_argument('--max_doc_len_bert',type=float,default= 400, help='max number for tokens per document for Bert model')
    # model struct
    parser.add_argument('--share_word_encoder', action="store_false", help='whether emotion and cause share the same underlying word encoder')
    parser.add_argument('--choose_emocate', default='',help="whether to predict emotion category")
    parser.add_argument('--use_x_v', default='', help='whether use video embedding')
    parser.add_argument('--use_x_a', default='', help='whether use audio embedding')
    parser.add_argument('--n_hidden', default=100, help='number of hidden unit')
    parser.add_argument('--n_class', default=2., help='number of distinct class')
    # Paths
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
    parser.add_argument("--keep_prob1", type=float,default=0.5,required=False,help="word embedding training dropout keep prob")
    parser.add_argument("--keep_prob2", type=float,default=1.0,required=False,help="softmax layer dropout keep prob")
    parser.add_argument("--keep_prob_v", type=float,default=0.5,required=False,help="training dropout keep prob for visual features")
    parser.add_argument("--keep_prob_a", type=float,default=0.5,required=False,help="training dropout keep prob for audio features")
    parser.add_argument("--emo",type=float,default=1.,help="loss weight of emotion ext.")
    parser.add_argument("--cause",type=float,default=1.0,help="loss weight for cause ext.")
    parser.add_argument("--bert_start_idx", type=float, default=20, help='bert para')
    parser.add_argument("--bert_end_idx", type=float, default=219, help='bert para')
    parser.add_argument("--bert_hidden_kb", type=float, default=0.9, help='keep prob for bert')
    parser.add_argument("--bert_attention_kb", type=float, default=0.7, help='keep prob for bert')

    all_args = parser.parse_known_args(args)[0]
    return all_args

def main():
    args = parse(sys.argv[1:])
    text_dir = os.path.join(args.input_dir, 'text')

    t_i = load_text_data(os.path.join('./../data','Subtask_2_1_train.json'))

if __name__ == '__main__':
    main()
