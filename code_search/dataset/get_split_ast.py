import os
import sys
import time
import json
import logging
import logging.config
import argparse
from collections import Counter
from multiprocessing import cpu_count, Pool
import gzip
import itertools
import math
import re
import nltk.stem
import copy
import sys
import random
from functools import partial
import pickle
import networkx as nx
from networkx.drawing.nx_pydot import read_dot
from spiral import ronin

sys.path.append("../")

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.DEBUG)
logging_config = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
            'datefmt': '%m/%d/%Y %H:%M:%S'}},
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout'}},
    'loggers': {'': {'handlers': ['default']}}
}
logging.config.dictConfig(logging_config)
logger = logging.getLogger(__name__)
import json
class Config:
    type_set = ["MODIFIER", "RETURN", "NAME", "TYPE", "UPDATE", "IN", "CASE", "COND"]



def add_classname_save_to_java_file(data_dir, method_id, method):
    java_filename = os.path.join(data_dir, str(method_id) + ".java")
    with open(java_filename, "w", encoding="utf-8") as f:
        f.write("class demo { \n %s  \n}"%method )

def read_json_file(filename):
    with open(filename, 'r') as fp:
        data = fp.readlines()
    if len(data) == 1:
        data = json.loads(data[0])
    else:
        data = [json.loads(line) for line in data]
    return data


def is_correct_parsed(data):
    if type(data) == dict:
        id = list(data.keys)[0]
    else:
        id = data
    dot_file_log = os.path.join(args.split_ast_files_dir, "%s.java.ast.err.log" % str(id))
    if os.path.getsize(dot_file_log):
        return False
    else:
        return True

# def load_split_ast(fid):
#     dot_file_log = os.path.join(args.split_ast_files_dir, "%d-AST.dot"%fid)
#     with open(dot_file_log,"r") as f:
#         codeBlock = json.load(f)
#     return codeBlock


def save_json_data(data_dir, filename, data):
    os.makedirs(data_dir, exist_ok=True)
    file_name = os.path.join(data_dir, filename)
    with open(file_name, 'w') as output:
        if type(data) == list:
            if type(data[0]) in [str, list, dict]:
                for item in data:
                    output.write(json.dumps(item))
                    output.write('\n')

            else:
                json.dump(data, output)
        elif type(data) == dict:
            json.dump(data, output)
        else:
            raise RuntimeError('Unsupported type: %s' % type(data))
    logger.info("saved dataset in " + file_name)

# get split ast 
def get_splitted_ast(fid):
    dot_file = os.path.join(args.split_ast_files_dir, "%d-AST.dot"%fid)
    ast_graph = nx.DiGraph(read_dot(dot_file))
    node_information = dict(ast_graph.nodes(True))
    node_num = len(node_information)
    # try:
    root_nodes, subtrees = get_subtrees(ast_graph)
    # except:
        # return None
    if not root_nodes:
        logger.info("root_nodes is none %d " % fid)
        return None
    try:
        converted_subtrees = convert_tree(root_nodes, subtrees, node_information)
        if  node_num > 500:
            converted_subtrees = converted_subtrees[:2]
        return converted_subtrees
    # except KeyError and IndexError:
    except KeyError and IndexError:
        logger.info("converted_subtrees %d " % fid)
        return None


def find_root(graph, child):
    parent = list(graph.predecessors(child))
    if len(parent) == 0:
        return child
    else:
        return find_root(graph, parent[0])


# Given a graph including some subtree
# return a list of root nodes  and  a list of subtrees
def get_subtrees(graph):
    root = []
    subtrees = {}
    all_nodes = copy.deepcopy(set(graph.nodes()))
    idx = 1
    number_nodes = len(all_nodes)
    while idx < number_nodes:
        node_idx = "n" + str(idx)
        node_idx = find_root(graph, node_idx)
        root.append(node_idx)
        tree = nx.dfs_tree(graph, node_idx)
        subtrees[node_idx] = nx.dfs_successors(graph, node_idx)
        subtree_nodes = set(tree.nodes())
        idx += len(subtree_nodes)
    return root, subtrees


def processing_node(s):
    s = s.strip()
    pat = r"'[\s\S]*'"
    s = re.sub(pat, "<STR>", s)
    res = [tok.lower() for tok in ronin.split(s) if tok]
    # return res
    return [[item] for item in res]


def format_node(node_label):
    node_label_tokens = node_label.split(": ")

    if node_label_tokens[0] == "INIT" and len(node_label_tokens) > 1:
        label_tokens = node_label_tokens[1].split(" ____ ")
        init_label_token = label_tokens[0].split()
        init_label_token.extend(label_tokens[1:])
        node_data = [[token.strip()] for token in init_label_token if token]
        node_data.insert(0, "INIT")
    elif node_label_tokens[0] in Config.type_set and len(node_label_tokens) > 1:
        label_tokens = node_label_tokens[1].split(" ____ ")
        node_data = [[token.strip()] for token in label_tokens if token]
        node_data.insert(0, node_label_tokens[0])
    else:
        label_tokens = node_label.split(" ____ ")
        if len(label_tokens) > 1:
            node_data = [[token] for token in label_tokens if token]
            node_data.insert(0, "STMT")
        else:
            node_data = [node_label]
    return node_data


def format_node_subtoken(node_label):
    node_label_tokens = node_label.split(": ")

    if node_label_tokens[0] == "INIT" and len(node_label_tokens) > 1:
        label_tokens = node_label_tokens[1].split(" ____ ")
        init_label_token = label_tokens[0].split()
        init_label_token.extend(label_tokens[1:])
        # node_data = [[token.strip()] for token in init_label_token if token]
        # node_data = [processing_node(token) for token in init_label_token if token][0]
        # node_data.insert(0, "INIT")
        node_data = ["INIT"]
        for token in init_label_token:
            if token:
                node_data.extend(processing_node(token))

    elif node_label_tokens[0] in Config.type_set and len(node_label_tokens) > 1:
        label_tokens = node_label_tokens[1].split(" ____ ")
        # node_data = [[token.strip()] for token in label_tokens if token]
        # node_data = [processing_node(token) for token in label_tokens if token][0]
        # node_data.insert(0, node_label_tokens[0])
        node_data = [node_label_tokens[0]]
        for token in label_tokens:
            if token:
                node_data.extend(processing_node(token))
    else:
        label_tokens = node_label.split(" ____ ")
        if len(label_tokens) > 1:
            # node_data = [[token] for token in label_tokens if token]
            # node_data = [processing_node(token) for token in label_tokens if token][0]
            # node_data.insert(0, "STMT")
            node_data = ["STMT"]
            for token in label_tokens:
                if token:
                    node_data.extend(processing_node(token))
        else:
            node_data = [node_label]
            # node_data = processing_node(node_label)
    return node_data


# trees : a -> b -> c; b->d; x->y->w,y->z;
# root: a, x
# subtrees_with_label [[a,[b,[c],[d]]],  [x,[y ,[w], [z]]]]
def convert_dict_to_list(root_node, tree, tree_with_label, node_info):
    children = tree[root_node]
    for r in children:
        if tree_with_label[0] == "ROOT":
            node = format_node(node_info[r]["label"][1:-1])
        else:
            node = format_node_subtoken(node_info[r]["label"][1:-1])
        # if len(node) > 1:
        #     tree_with_label.extend(node)
        # else:
        tree_with_label.append(node)
        if r in tree.keys():
            convert_dict_to_list(r, tree, tree_with_label[-1], node_info)
        else:
            continue


# converting some trees  to  the list
# for example
# trees : a -> b -> c; b->d; x->y->w,y->z;
# root: a, x
# subtrees_with_label [[a,[b,[c],[d]]],  [x,[y ,[w], [z]]]]
def convert_tree(root, trees, node_data):
    subtrees_with_node_label = []
    for r in root:
        subtree = trees[r]
        node_label = node_data[r]["label"][1:-1]
        node_label_token = node_label.split(":")
        if len(node_label_token) > 1:
            subtrees_with_node_label.append(["ROOT"])
        else:
            subtrees_with_node_label.append([node_label])

        convert_dict_to_list(r, subtree, subtrees_with_node_label[-1], node_data)

    return subtrees_with_node_label


def tree_to_seq_with_parent_position(trees, parent_position, seq, position):
    if trees[0] == 'METHOD_BODY':
        pass
    else:
        seq.append(trees[0])
        position.append(parent_position)
        parent_position = len(position)
    for i in range(1, len(trees)):
        node = trees[i]
        tree_to_seq_with_parent_position(node, parent_position, seq, position)


def postorder_traversal(trees, seq):
    for i in range(1, len(trees)):
        node = trees[i]
        if len(node) == 1:
            seq.append(node[0])
        else:
            postorder_traversal(node, seq)
    if type(trees[0]) == str:
        seq.append(trees[0])


def process_main_body(main_body):
    position = []
    seq = []
    parent_position = 0
    tree_to_seq_with_parent_position(main_body, parent_position, seq, position)
    return seq, position


def process_splitted_tree(subtrees):
    # subtrees = sliced_AST
    subtrees_list = []
    subtrees_root_node_list = []
    # trees_root_node_list =  subtrees_root_node_list

    for subtree in subtrees:
        subtrees_root_node_list.append(subtree[0])
        seq = []
        postorder_traversal(subtree, seq)
        subtrees_list.append(seq)
    return subtrees_list, subtrees_root_node_list


def get_parent_of_node(main_body_seq, main_body_parent, root_list, subtrees_list):
    tree_parent = main_body_parent[:2]
    main_body_point = 2
    subtrees_root_node_list_point = 2
    try:
        for root_node in root_list[2:]:

            if main_body_point >= len(main_body_seq):
                break
            if root_node == main_body_seq[main_body_point]:
                tree_parent.append(main_body_parent[main_body_point])
                main_body_point += 1
                subtrees_root_node_list_point += 1
            if main_body_point < len(main_body_seq):
                if main_body_seq[main_body_point] == "STATIC-BLOCK":
                    main_body_point += 1
                else:
                    main_boy_node_index = root_list.index(main_body_seq[main_body_point],
                                                          subtrees_root_node_list_point)
            if root_node[:7] == "NESTED_":
                subtrees_root_node_list_point += 1
                if root_node in subtrees_list[subtrees_root_node_list_point][:-1]:
                    tree_parent.append(subtrees_root_node_list_point + 1)
                else:
                    tree_parent.append(main_boy_node_index + 1)
        return tree_parent
    except Exception as e:
        # print("wrong fid: ", fid)
        # print("main_body_seq: ", main_body_seq)
        # print("main_body_parent: ", main_body_parent)
        # print("subtrees_root_node_list: ", root_list)
        # Config.wrong_cnt += 1
        # print("wrong_cnt: ", Config.wrong_cnt)
        print(str(e))
        return None


def get_tree_format(tree_parent):
    tree_parent_index = [p - 1 for p in tree_parent]
    rebuild_tree = {}
    for i, parent in enumerate(tree_parent_index[1:]):
        if parent in rebuild_tree.keys():
            rebuild_tree[parent].append(i + 1)
        else:
            rebuild_tree[parent] = [i + 1]
    return rebuild_tree


def rebuild_structure_tree(sliced_AST):
    # main body
    main_body_seq, main_body_parent = process_main_body(sliced_AST[0])

    # sliced AST
    subtrees_list, subtrees_root_node_list = process_splitted_tree(sliced_AST)

    # root nodes set -> sliced_tree_parent.
    sliced_tree_parent = get_parent_of_node(main_body_seq, main_body_parent, subtrees_root_node_list, subtrees_list)

    # rebuild tree
    if sliced_tree_parent:
        rebuild_tree = get_tree_format(sliced_tree_parent)
        return rebuild_tree
    else:
        return None

def process(args):
    # read the data and load the  source code
    ## train code 
    filename="java/train.jsonl"
    train_jsonl = read_json_file(filename)
    if args.debug:
        train_jsonl = train_jsonl[:args.n_debug_samples]
    train_code  = {fid:item["code"] for fid, item in enumerate(train_jsonl)}
    all_train_length = len(train_code)

    filename ="java/codebase.jsonl"
    codebase_jsonl = read_json_file(filename)
    if args.debug:
        codebase_jsonl = codebase_jsonl[:args.n_debug_samples]
    codebase_code  = {fid+all_train_length :item["code"] for fid, item in enumerate(codebase_jsonl)}

    # logger.info("***** loading data *****")
    # filename = os.path.join(args.data_dir, "codesearchnet.pkl" )
    # codesearchnet = pickle.load(open( filename , "rb"))
    # methods_code = {part: {fid: item['code'] for fid, item in codesearchnet[part].items()} for part in codesearchnet}
    methods_code = {"train":train_code, "codebase":codebase_code}

    # saving the src to java file
    logger.info("***** saving the src to java file  *****")
    os.makedirs(args.java_files_dir, exist_ok=True)
    # for partition in ["train", "valid", "test"]:
    for partition in methods_code.keys():
        codes = methods_code[partition]
        _ = [add_classname_save_to_java_file(args.java_files_dir, idx, code) for idx, code in
                codes.items()]
    # exit(1)
    # using jar to parser 
    logger.info("***** using jar to parser *****")
    # command_line = r" java -jar split_code_block.jar -outdir %s -ast -node_level block -lang java -format json %s"%(args.split_ast_files_dir, args.java_files_dir)
    command_line = r" java -jar progex_s.jar  -outdir %s -ast -lang java -format dot  -node_level block   %s"%(args.split_ast_files_dir, args.java_files_dir)
    os.system(command_line) #todo

    # get correct fid
    logger.info("*****  getting correct fid *****")
    cores = cpu_count()
    correct_fids = {}
    for part in methods_code.keys():
        pool = Pool(cores)
        results = pool.map(is_correct_parsed, list(methods_code[part]))
        pool.close()
        pool.join()
        correct_fids[part] = [fid for i, fid in enumerate(list(methods_code[part])) if results[i]]
        logger.info("%s Done" % part)

    # load the split ast and save to json
    logger.info("*****  getting split ast *****")
    cores = cpu_count()
    split_ast_dict  = {}
    results =[]
    for part in methods_code.keys():
        pool = Pool(cores)
        results = pool.map(get_splitted_ast, correct_fids[part])
        pool.close()
        pool.join()
        # for fid in correct_fids[part]:
        #     results.append(get_splitted_ast(fid))

        split_ast_dict [part] = { fid:results[i] for i, fid in enumerate(correct_fids[part]) }
        logger.info("%s Done" % part)
    filename = "split_ast.jsonl"
    output_dir = os.path.join(args.output_dir, "tmp")
    save_json_data(output_dir, filename, split_ast_dict)

    # get rebuild ast
    rebuild_tree_part= {}
    cores = cpu_count()
    for part in split_ast_dict.keys():
        pool = Pool(cores)
        results = pool.map(rebuild_structure_tree, list(split_ast_dict[part].values()))
        pool.close()
        pool.join()
        # correct_rebuild_tree_fid[part] = [fid for i, fid in enumerate(sliced_AST_list[part].keys()) if
        #                                     results[i]]
        rebuild_tree_part[part] = {fid: results[i] for i, fid in enumerate(split_ast_dict[part].keys()) if results[i]}
    filename = "rebuild_ast.jsonl"
    output_dir = os.path.join(args.output_dir, "tmp")
    save_json_data(output_dir, filename, rebuild_tree_part)

    # 
    # for fid, item in enumerate(train_jsonl):
    for fid, item in enumerate(train_jsonl):
        try:
            item["split_ast"] = split_ast_dict["train"][fid]
            item["rebuild_ast"] = rebuild_tree_part["train"][fid]
        except KeyError:
            item["split_ast"],item["rebuild_ast"] = [],[]
    filename = "train_add_ast.jsonl"
    save_json_data(args.output_dir, filename, train_jsonl)

    for fid, item in enumerate(codebase_jsonl):
        try:
            item["split_ast"] = split_ast_dict["codebase"][fid+all_train_length]
            item["rebuild_ast"] = rebuild_tree_part["codebase"][fid+all_train_length]
        except KeyError:
            item["split_ast"],item["rebuild_ast"] = [],[]
    filename = "codebase_add_ast.jsonl"
    save_json_data(args.output_dir, filename, codebase_jsonl)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='debug mode', required=False)
    parser.add_argument('--n_debug_samples', type=int, default=100, required=False)
    parser.add_argument('-data_dir', type=str, default="/home/v-ensh/sciteamdrive2/v-ensh/workspace/SE_Data/data_about_task/codesearch/csn/mini")
    parser.add_argument('-java_files_dir', type=str, default="/home/t-enshengshi/codesearchnet/java_files/mini")
    parser.add_argument('-split_ast_files_dir', type=str, default="/home/t-enshengshi/codesearchnet/split_ast/mini")
    parser.add_argument('-output_dir', type=str, default="java")
    args = parser.parse_args()
    return args


def main(): 
    logger.info(args)
    process(args)


if __name__ == '__main__':
    args = parse_args()
    logger.info(args)
    main()