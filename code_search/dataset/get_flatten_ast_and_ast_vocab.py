#!/usr/bin/env python
# !-*-coding:utf-8 -*-

import os
import time
import sys
from multiprocessing import cpu_count, Pool
from collections import Counter

# sys.path.append("../util")
# from Config import Config as cf
# from LoggerUtil import set_logger, debug_logger
# from DataUtil import read_pickle_data, tree_to_index, save_pickle_data, make_directory, time_format

sys.path.append("../")
from util import read_json_file,time_format,save_pickle_data,save_json_data,array_split
import logging
logger = logging.getLogger(__name__)
#set log
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )

def preOrderTraverse(tree, sequence):
    if not tree:
        return None
    #     print(tree[0])
    sequence.append(tree[0])
    for subtree in tree[1:]:
        preOrderTraverse(subtree, sequence)


def get_splitted_ast_sequence(tree):
    sequence = []
    for subtree in tree:
        preOrderTraverse(subtree, sequence)
    return sequence


def count_word_parallel(word_list):
    cores = cpu_count()
    pool = Pool(cores)
    word_split = array_split(word_list, cores)
    word_counts = pool.map(Counter, word_split)
    result = Counter()
    for wc in word_counts:
        result += wc
    pool.close()
    pool.join()
    return dict(result.most_common())  # return the dict sorted by frequency reversely.


if __name__ == '__main__':

    filename = "java/train_add_ast.jsonl"
    train_data = read_json_file(filename)
    asts_tree = [item["split_ast"] for item in train_data]

    # get flaten ast   
    logger.info("Getting flatten ast") 
    cores = cpu_count()
    pool = Pool(cores)
    results = pool.map(get_splitted_ast_sequence, asts_tree)
    pool.close()
    pool.join()
    asts_sequence_corpus = [ results[fid] for fid  in range(len(results)) if results[fid]]

    # get all token 
    logger.info("Getting all token") 
    all_tokens = []
    for one_ast_sequence in asts_sequence_corpus:
        all_tokens.extend(one_ast_sequence)
    
    # get vocabulary
    logger.info("Getting vocabulary") 
    ast_vocab = count_word_parallel(all_tokens)
    filename = "java/ast_vocab.jsonl"
    save_json_data("java",  "ast_vocab.jsonl", ast_vocab  )