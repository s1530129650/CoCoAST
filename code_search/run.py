# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

import argparse
import logging
import os
import pickle
import random
import torch
import json
import copy
import numpy as np
from zmq import device
from model import Model, CAST
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                  RobertaConfig, RobertaModel, RobertaTokenizer)
from util import  cal_r1_r5_r10, read_json_file

logger = logging.getLogger(__name__)

from tqdm import tqdm, trange
import multiprocessing
from multiprocessing import cpu_count
cpu_cont = cpu_count()
torch.multiprocessing.set_sharing_strategy('file_system')
from parser import DFG_python,DFG_java,DFG_ruby,DFG_go,DFG_php,DFG_javascript
from parser import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_index)
from tree_sitter import Language, Parser
dfg_function={
    'python':DFG_python,
    'java':DFG_java,
    'ruby':DFG_ruby,
    'go':DFG_go,
    'php':DFG_php,
    'javascript':DFG_javascript
}

#load parsers
parsers={}        
for lang in dfg_function:
    LANGUAGE = Language('parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE) 
    parser = [parser,dfg_function[lang]]    
    parsers[lang]= parser
    
    
#remove comments, tokenize code and extract dataflow                                        
def extract_dataflow(code, parser,lang):
    #remove comments
    try:
        code=remove_comments_and_docstrings(code,lang)
    except:
        pass    
    #obtain dataflow
    if lang=="php":
        code="<?php"+code+"?>"    
    try:
        tree = parser[0].parse(bytes(code,'utf8'))    
        root_node = tree.root_node  
        tokens_index=tree_to_token_index(root_node)     
        code=code.split('\n')
        code_tokens=[index_to_code_token(x,code) for x in tokens_index]  
        index_to_code={}
        for idx,(index,code) in enumerate(zip(tokens_index,code_tokens)):
            index_to_code[index]=(idx,code)  
        try:
            DFG,_=parser[1](root_node,index_to_code,{}) 
        except:
            DFG=[]
        DFG=sorted(DFG,key=lambda x:x[1])
        indexs=set()
        for d in DFG:
            if len(d[-1])!=0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG=[]
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg=new_DFG
    except:
        dfg=[]
    return code_tokens,dfg

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 code_tokens,
                 code_ids,               
                 nl_tokens,
                 nl_ids,
                 split_ast_list,
                 split_ast_list_ids,
                 rebuild_tree,
                 url,

    ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids

        self.split_ast_list = split_ast_list
        self.split_ast_list_ids = split_ast_list_ids
       
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids

        self.rebuild_tree = rebuild_tree

        self.url=url

class ASTTokenizer(object):
    def __init__(self,tokenizer,  args ):
        self.bos_token = tokenizer.bos_token
        self.eos_token = tokenizer.eos_token
        self.unk_token  = tokenizer.unk_token 
        self.sep_token  = tokenizer.sep_token 
        self.pad_token = tokenizer.pad_token
        self.cls_token = tokenizer.cls_token
        # self.mask_token = tokenizer.mask_token 
        self.additional_special_tokens = tokenizer.additional_special_tokens

        self.bos_token_id = 0
        self.cls_token_id = 0
        self.pad_token_id = 1
        self.eos_token_id = 2
        self.sep_token_id  = 2
        self.unk_token_id  = 3
        
        self.ast_vocab_file = args.ast_vocab_file
        self.ast_vocab_size = args.ast_vocab_size
        self.w2i = {self.bos_token: self.bos_token_id,self.pad_token: self.pad_token_id,self.eos_token: self.eos_token_id }
        self.i2w = {self.bos_token_id: self.bos_token,self.pad_token_id: self.pad_token,self.eos_token_id: self.eos_token }
        self.build_vocab()
    
    def build_vocab(self, ):
        start_id = 4
        word_count = read_json_file(self.ast_vocab_file )
        # word_count_ord[i][0] -> word, word_count_ord[i][1] -> count
        word_count_ord = sorted(word_count.items(), key=lambda item: item[1], reverse=True)

        if self.ast_vocab_size  > 0:
            if self.ast_vocab_size  < len(word_count):
                size = self.ast_vocab_size 
            else:
                size = len(word_count)
            print("vocab_size (exclude special tokens) %d" % size)

        else:
            size = len(word_count_ord)
            print("use all tokens %d " % size)

        for i in range(start_id+1,size,1):
            self.w2i[word_count_ord[i][0]] = i 
            self.i2w[i] = word_count_ord[i][0]

    def tree2idx(self, trees,):
        
        for i in range(len(trees)):
            try:
                node = trees[i]
            except:
                pass
            if type(node) == str:
                trees[i] = self.w2i[node] if node in self.w2i else self.pad_token_id
            else:
                self.tree2idx(node)




def tokenizer_source_code(code, parser,lang):
    #remove comments
    try:
        code=remove_comments_and_docstrings(code,lang)
    except:
        pass    
    #obtain dataflow
    if lang=="php":
        code="<?php"+code+"?>"    
    try:
        tree = parser[0].parse(bytes(code,'utf8'))    
        root_node = tree.root_node  
        tokens_index=tree_to_token_index(root_node)     
        code=code.split('\n')
        code_tokens=[index_to_code_token(x,code) for x in tokens_index]  
    except:
        dfg=[]
    return code_tokens       
        
def convert_examples_to_features(item):
    js,tokenizer,ast_tokenizer,args=item
    #code
    parser=parsers[args.lang]
    #extract data flow
    # code
    code_tokens=tokenizer_source_code(js['original_string'],parser,args.lang)
    code_tokens=" ".join(code_tokens[:args.code_length-2])
    code_tokens=tokenizer.tokenize(code_tokens)[:args.code_length-2]
    code_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    code_ids =  tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = args.code_length - len(code_ids)
    code_ids+=[tokenizer.pad_token_id]*padding_length   

    #nl
    nl=' '.join(js['docstring_tokens'])
    nl_tokens=tokenizer.tokenize(nl)[:args.nl_length-2]
    nl_tokens =[tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]
    nl_ids =  tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.nl_length - len(nl_ids)
    nl_ids+=[tokenizer.pad_token_id]*padding_length    

    # ast
    split_ast_list = None
    split_ast_list_ids= None
    rebuild_tree = None
    if "split_ast" in js.keys():
        split_ast_list = js["split_ast"]
        if len(split_ast_list ) < 1:
            split_ast_list = [[ast_tokenizer.pad_token]] 
            split_ast_list_ids = [[ast_tokenizer.pad_token_id],[ast_tokenizer.pad_token_id]] 
            rebuild_tree = {0:[1]}
        else:
            split_ast_list_ids = copy.deepcopy(split_ast_list)
            ast_tokenizer.tree2idx(split_ast_list_ids)
            if "rebuild_ast" in js.keys():
                rebuild_tree = js["rebuild_ast"]
            rebuild_tree =  {int(k):v for k,v in rebuild_tree.items() }


    return InputFeatures(code_tokens,code_ids,nl_tokens,nl_ids, split_ast_list,split_ast_list_ids , rebuild_tree, js['url'])
    # return InputFeatures(code_tokens,code_ids,position_idx,dfg_to_code,dfg_to_dfg,nl_tokens,nl_ids,js['url'])

class TextDataset(Dataset):
    def __init__(self, tokenizer,ast_tokenizer, args, file_path=None,pool=None):
        self.args=args
        prefix=file_path.split('/')[-1][:-6]
        cache_file=args.output_dir+'/'+prefix+'.pkl'
        n_debug_samples = args.n_debug_samples
        if 'codebase' in file_path:
            n_debug_samples = 100
        if os.path.exists(cache_file):
            self.examples=pickle.load(open(cache_file,'rb'))
            if args.debug:
                self.examples= self.examples[:n_debug_samples]
        else:
            self.examples = []
            data=[]
            if args.debug:
                with open(file_path, encoding="utf-8") as f:
                    for line in f:
                        line=line.strip()
                        js=json.loads(line)
                        data.append((js,tokenizer,ast_tokenizer,args))
                        if len(data) >= n_debug_samples:
                            break
            else:
                with open(file_path) as f:
                    for line in f:
                        line=line.strip()
                        js=json.loads(line)
                        data.append((js,tokenizer,ast_tokenizer,args))
            self.examples=pool.map(convert_examples_to_features, tqdm(data,total=len(data)))
            # pickle.dump(self.examples,open(cache_file,'wb'))
            
        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:1]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("code_tokens: {}".format([x.replace('\u0120','_') for x in example.code_tokens]))
                logger.info("code_ids: {}".format(' '.join(map(str, example.code_ids))))
                # logger.info("position_idx: {}".format(example.position_idx))
                # logger.info("dfg_to_code: {}".format(' '.join(map(str, example.dfg_to_code))))
                # logger.info("dfg_to_dfg: {}".format(' '.join(map(str, example.dfg_to_dfg))))                
                logger.info("nl_tokens: {}".format([x.replace('\u0120','_') for x in example.nl_tokens]))
                logger.info("nl_ids: {}".format(' '.join(map(str, example.nl_ids))))          
                
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item): 
        # #calculate graph-guided masked function
        # attn_mask=np.zeros((self.args.code_length+self.args.data_flow_length,
        #                     self.args.code_length+self.args.data_flow_length),dtype=np.bool)
        # #calculate begin index of node and max length of input
        # node_index=sum([i>1 for i in self.examples[item].position_idx])
        # max_length=sum([i!=1 for i in self.examples[item].position_idx])
        # #sequence can attend to sequence
        # attn_mask[:node_index,:node_index]=True
        # #special tokens attend to all tokens
        # for idx,i in enumerate(self.examples[item].code_ids):
        #     if i in [0,2]:
        #         attn_mask[idx,:max_length]=True
        # #nodes attend to code tokens that are identified from
        # for idx,(a,b) in enumerate(self.examples[item].dfg_to_code):
        #     if a<node_index and b<node_index:
        #         attn_mask[idx+node_index,a:b]=True
        #         attn_mask[a:b,idx+node_index]=True
        # #nodes attend to adjacent nodes 
        # for idx,nodes in enumerate(self.examples[item].dfg_to_dfg):
        #     for a in nodes:
        #         if a+node_index<len(self.examples[item].position_idx):
        #             attn_mask[idx+node_index,a+node_index]=True  
                    
        return (torch.tensor(self.examples[item].code_ids),
                self.examples[item].split_ast_list_ids,
                self.examples[item].rebuild_tree,
                torch.tensor(self.examples[item].nl_ids)
               )

            

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def collate_fn(batch):
        return batch

def train(args, model, tokenizer,ast_tokenizer, pool):
    """ Train the model """
    #get training dataset
    train_dataset=TextDataset(tokenizer,ast_tokenizer, args, args.train_data_file, pool)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=0,collate_fn=collate_fn,drop_last=True)
    
    #get optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,num_training_steps=len(train_dataloader)*args.num_train_epochs)
    
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//args.n_gpu)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", len(train_dataloader)*args.num_train_epochs)
    
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()
    
    model.train()
    tr_num,tr_loss,best_mrr=0,0,-1 
    for idx in range(args.num_train_epochs): 
        for step,batch in enumerate(train_dataloader):
            #get inputs
            # code_inputs = batch[0].to(args.device)  
            # attn_mask = batch[1].to(args.device)
            # position_idx = batch[2].to(args.device)
            # nl_inputs = batch[3].to(args.device)

            # code 
            code_inputs = [item[0].numpy()  for item in batch]
            code_inputs = torch.tensor(code_inputs).to(args.device)

            # ast 
            split_ast = [item[1] for item in batch]

            # rebuild tree
            rebuild_tree = [item[2] for item in batch]

            # query
            nl_inputs  = [item[-1].numpy()  for item in batch]
            nl_inputs  = torch.tensor(nl_inputs ).to(args.device)

            #get code and nl vectors
            # code_vec = model(code_inputs=code_inputs,attn_mask=attn_mask,position_idx=position_idx)
            code_vec = model(code_inputs=code_inputs, split_ast=split_ast, rebuild_tree=rebuild_tree)
            nl_vec = model(nl_inputs=nl_inputs)
            
            #calculate scores and loss
            scores=torch.einsum("ab,cb->ac",nl_vec,code_vec)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(scores, torch.arange(code_inputs.size(0), device=scores.device))
            
            #report loss
            tr_loss += loss.item()
            tr_num+=1
            if (step+1)% args.eval_frequency ==0:
                logger.info("epoch {} step {} loss {}".format(idx,step+1,round(tr_loss/tr_num,5)))
                tr_loss=0
                tr_num=0
            
            #backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step() 
            
        #evaluate    
        results = evaluate(args, model, tokenizer, ast_tokenizer, args.eval_data_file, pool, eval_when_training=True)
        for key, value in results.items():
            logger.info("  %s = %s", key, round(value,3))    
            
        #save best model
        if results['eval_mrr']>best_mrr:
            best_mrr=results['eval_mrr']
            logger.info("  "+"*"*20)  
            logger.info("  Best mrr:%s",round(best_mrr,3))
            logger.info("  "+"*"*20)                          

            checkpoint_prefix = 'checkpoint-best-mrr'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)                        
            model_to_save = model.module if hasattr(model,'module') else model
            output_dir = os.path.join(output_dir, '{}'.format('model.bin')) 
            torch.save(model_to_save.state_dict(), output_dir)
            logger.info("Saving model checkpoint to %s", output_dir)


def evaluate(args, model, tokenizer,ast_tokenizer,file_name,pool, eval_when_training=False):
    query_dataset = TextDataset(tokenizer, ast_tokenizer, args, file_name, pool)
    query_sampler = SequentialSampler(query_dataset)
    query_dataloader = DataLoader(query_dataset, sampler=query_sampler, batch_size=args.eval_batch_size,num_workers=0,collate_fn=collate_fn)
    
    code_dataset = TextDataset(tokenizer, ast_tokenizer, args, args.codebase_file, pool)
    code_sampler = SequentialSampler(code_dataset)
    code_dataloader = DataLoader(code_dataset, sampler=code_sampler, batch_size=args.eval_batch_size,num_workers=0,collate_fn=collate_fn)    

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num queries = %d", len(query_dataset))
    logger.info("  Num codes = %d", len(code_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    
    model.eval()
    code_vecs=[] 
    nl_vecs=[]
    for batch in query_dataloader:  
        # nl_inputs = batch[3].to(args.device)
        nl_inputs  = [item[-1].numpy()  for item in batch]
        nl_inputs  = torch.tensor(nl_inputs ).to(args.device)
        with torch.no_grad():
            nl_vec = model(nl_inputs=nl_inputs) 
            nl_vecs.append(nl_vec.cpu().numpy()) 

    for batch in code_dataloader:
        # code_inputs = batch[0].to(args.device)    
        # attn_mask = batch[1].to(args.device)
        # position_idx =batch[2].to(args.device)
        # code 
        code_inputs = [item[0].numpy()  for item in batch]
        code_inputs = torch.tensor(code_inputs).to(args.device)
        if len(batch) < args.eval_batch_size:
            with torch.no_grad():
                code_vec = model(nl_inputs=code_inputs) 
        else:


            # ast 
            split_ast = [item[1] for item in batch]

            # rebuild tree
            rebuild_tree = [item[2] for item in batch]
            with torch.no_grad():
                # code_vec= model(code_inputs=code_inputs, attn_mask=attn_mask,position_idx=position_idx)
                code_vec = model(code_inputs=code_inputs, split_ast=split_ast, rebuild_tree=rebuild_tree)
        code_vecs.append(code_vec.cpu().numpy())  
    model.train()    
    code_vecs=np.concatenate(code_vecs,0)
    nl_vecs=np.concatenate(nl_vecs,0)

    scores=np.matmul(nl_vecs,code_vecs.T)
    
    sort_ids=np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1]    
    
    nl_urls=[]
    code_urls=[]
    for example in query_dataset.examples:
        nl_urls.append(example.url)
        
    for example in code_dataset.examples:
        code_urls.append(example.url)
        
    ranks=[]
    for url, sort_id in zip(nl_urls,sort_ids):
        rank=0
        find=False
        for idx in sort_id[:1000]:
            if find is False:
                rank+=1
            if code_urls[idx]==url:
                find=True
        if find:
            ranks.append(1/rank)
        else:
            ranks.append(0)
    result = cal_r1_r5_r10(ranks)
    result["eval_mrr"]  = float(np.mean(ranks))
    # result = {
    #     "eval_mrr":float(np.mean(ranks))
    # }

    return result

                        
                        
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument('--debug', action='store_true', help='debug mode', required=False)
    parser.add_argument('--n_debug_samples', type=int, default=100, required=False)
    parser.add_argument("--eval_frequency", default=1, type=int, required=False)
    parser.add_argument('--use_pre_trained', action='store_true', help='debug mode', required=False)

    parser.add_argument("--train_data_file", default="dataset/java/train_add_ast.jsonl", type=str, required=False,
                        help="The input training data file (a json file).")
    parser.add_argument("--ast_vocab_file", default="dataset/java/ast_vocab.jsonl", type=str, required=False,
                        help="The vocabulary of the ast ast_vocab.jsonl")
    parser.add_argument("--output_dir", default="saved_models/graphcodebert/tmp", type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_data_file", default="dataset/java/valid.jsonl", type=str,
                        help="An optional input evaluation data file to evaluate the MRR(a jsonl file).")
    parser.add_argument("--test_data_file", default="dataset/java/test.jsonl", type=str,
                        help="An optional input test data file to test the MRR(a josnl file).")
    parser.add_argument("--codebase_file", default="dataset/java/codebase_add_ast.jsonl", type=str,
                        help="An optional input test data file to codebase (a jsonl file).")  
    
    parser.add_argument("--lang", default="java", type=str,
                        help="language.")  
    
    parser.add_argument("--model_name_or_path", default="microsoft/graphcodebert-base", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="microsoft/graphcodebert-base", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="microsoft/graphcodebert-base", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    
    parser.add_argument("--nl_length", default=20, type=int,
                        help="Optional NL input sequence length after tokenization.")    
    parser.add_argument("--code_length", default=50, type=int,
                        help="Optional Code input sequence length after tokenization.") 
    parser.add_argument("--data_flow_length", default=10, type=int,
                        help="Optional Data Flow input sequence length after tokenization.") 
    parser.add_argument("--ast_vocab_size", default=1000, type=int,
                        help="The size of the ast vocabulary") 
    
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")  
    
    parser.add_argument('--hidden_size', type=int, default=512, required=False)
    parser.add_argument('--intermediate_size', type=int, default=1024, required=False)
    parser.add_argument('--num_attention_heads', type=int, default=8, required=False)
    parser.add_argument('--num_hidden_layers', type=int, default=6, required=False)

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=2, type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument('--seed', type=int, default=3407,
                        help="random seed for initialization")
    
    pool = multiprocessing.Pool(cpu_cont)
    
    #print arguments
    args = parser.parse_args()
    
    #set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )
    #set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s",device, args.n_gpu)
    
    # Set seed
    set_seed(args.seed)

    #build model
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    

    # config
    if args.use_pre_trained:
        config = RobertaConfig.from_pretrained(args.model_name_or_path)
        transformer_encoder = RobertaModel.from_pretrained(args.model_name_or_path)    
        args.hidden_size = 768
    else:
        ast_tokenizer = ASTTokenizer(tokenizer, args)
        config = RobertaConfig()
        config.hidden_size = args.hidden_size
        config.intermediate_size = args.intermediate_size
        config.num_attention_heads = args.num_attention_heads
        config.num_hidden_layers = args.num_hidden_layers
        config.vocab_size = tokenizer.vocab_size 
        transformer_encoder = RobertaModel(config) 
    model=CAST(transformer_encoder,args)
    logger.info("Training/evaluation parameters %s", args)
    logger.info(model.model_parameters())
    model.to(args.device)
    
    # Training
    if args.do_train:
        train(args, model, tokenizer,ast_tokenizer, pool)

    # Evaluation
    results = {}
    # if args.do_eval:
    #     checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
    #     output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
    #     model.load_state_dict(torch.load(output_dir),strict=False)      
    #     model.to(args.device)
    #     result=evaluate(args, model, tokenizer,args.eval_data_file, pool)
    #     logger.info("***** Eval results *****")
    #     for key in sorted(result.keys()):
    #         logger.info("  %s = %s", key, str(round(result[key],3)))
            
    if args.do_test:
        checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir),strict=False)      
        model.to(args.device)
        result=evaluate(args, model, tokenizer,ast_tokenizer, args.test_data_file, pool)
        logger.info("***** Eval results *****")
        latex_output="&"
        for key in result.keys():
            logger.info("  %s = %s", key, str(round(result[key],3)))
            latex_output += "& %s"%(str(round(result[key],3)))
        logger.info(latex_output)
    return results


if __name__ == "__main__":
    main()


