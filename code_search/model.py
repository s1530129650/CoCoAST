# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch.nn as nn
import torch   
from prettytable import PrettyTable
from torch.nn.modules.activation import Tanh
from models.RvNNRvNNASTCodeAttn import BatchASTEncoder,ASTNNEncoder
class BaseModel(nn.Module): 
    def __init__(self, ):
        super().__init__()
        
    def model_parameters(self):
        table = PrettyTable()
        table.field_names = ["Layer Name", "Output Shape", "Param #"]
        table.align["Layer Name"] = "l"
        table.align["Output Shape"] = "r"
        table.align["Param #"] = "r"
        for name, parameters in self.named_parameters():
            if parameters.requires_grad:
                table.add_row([name, str(list(parameters.shape)), parameters.numel()])
        return table

class Model( BaseModel):   
    def __init__(self, encoder):
        super(Model, self).__init__()
        self.encoder = encoder
      
    def forward(self, code_inputs=None, attn_mask=None,position_idx=None, nl_inputs=None): 
        if code_inputs is not None:
            nodes_mask=position_idx.eq(0)
            token_mask=position_idx.ge(2)        
            inputs_embeddings=self.encoder.embeddings.word_embeddings(code_inputs)
            nodes_to_token_mask=nodes_mask[:,:,None]&token_mask[:,None,:]&attn_mask
            nodes_to_token_mask=nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]
            avg_embeddings=torch.einsum("abc,acd->abd",nodes_to_token_mask,inputs_embeddings)
            inputs_embeddings=inputs_embeddings*(~nodes_mask)[:,:,None]+avg_embeddings*nodes_mask[:,:,None]    
            return self.encoder(inputs_embeds=inputs_embeddings,attention_mask=attn_mask,position_ids=position_idx)[1]
        else:
            return self.encoder(nl_inputs,attention_mask=nl_inputs.ne(1))[1]
            

class BaselineModel( BaseModel):   
    def __init__(self, encoder):
        super(BaselineModel, self).__init__()
        self.encoder = encoder
      
    def forward(self, code_inputs=None, attn_mask=None,position_idx=None, nl_inputs=None): 
        if code_inputs is not None:
            return self.encoder(code_inputs,attention_mask=code_inputs.ne(1))[1]
        else:
            return self.encoder(nl_inputs,attention_mask=nl_inputs.ne(1))[1]

class POSModel( BaseModel):   
    def __init__(self, encoder):
        super(POSModel, self).__init__()
        self.encoder = encoder
      
    def forward(self, inputs=None, position_idx=None): 
        if position_idx is not None:
            return self.encoder(inputs,attention_mask=inputs.ne(1),position_ids=position_idx)[1]
        else:
            return self.encoder(inputs,attention_mask=inputs.ne(1))[1]
            

class CAST( BaseModel):   
    def __init__(self, transformer_encoder,args):
        super(CAST, self).__init__()
        self.encoder = transformer_encoder
        # self.ast_encoder= 
        self.hidden_size = args.hidden_size
        use_gpu = True if torch.cuda.is_available() else False
        self.astEncoder = BatchASTEncoder(args.hidden_size, args.ast_vocab_size, args.hidden_size,
                                          args.train_batch_size//args.n_gpu, use_gpu=use_gpu)
        self.mlp = nn.Sequential(nn.Linear(self.hidden_size*2,self.hidden_size)
        )
    def forward(self, code_inputs=None, split_ast=None, rebuild_tree=None, nl_inputs=None): 
        if (code_inputs is not None) and (split_ast is not None):
            _, split_full_tree_embedding = self.astEncoder(split_ast, rebuild_tree)
            # split_full_tree_embedding: [bs, hid_size]
            code_token_embedding = self.encoder(code_inputs, attention_mask=code_inputs.ne(1))[1]
            # code_token_embedding: [bs, hid_size]
            return self.mlp( torch.cat((split_full_tree_embedding, code_token_embedding), 1)  )
            # [bs, hid_size]
            
        else:
            return self.encoder(nl_inputs, attention_mask=nl_inputs.ne(1))[1]
            # [bs, hid_size]

class ASTNN( BaseModel):   
    def __init__(self, transformer_encoder,args):
        super(ASTNN, self).__init__()
        self.encoder = transformer_encoder
        # self.ast_encoder= 
        self.hidden_size = args.hidden_size
        use_gpu = True if torch.cuda.is_available() else False
        self.astEncoder = ASTNNEncoder(args.hidden_size, args.ast_vocab_size, args.hidden_size,
                                          args.train_batch_size//args.n_gpu, use_gpu=use_gpu)
        self.mlp = nn.Sequential(nn.Linear(self.hidden_size*2,self.hidden_size)
        )
    def forward(self, code_inputs=None, split_ast=None, rebuild_tree=None, nl_inputs=None): 
        if (code_inputs is not None) and (split_ast is not None):
            _, split_full_tree_embedding = self.astEncoder(split_ast, rebuild_tree)
            # split_full_tree_embedding: [bs, hid_size]
            code_token_embedding = self.encoder(code_inputs, attention_mask=code_inputs.ne(1))[1]
            # code_token_embedding: [bs, hid_size]
            return self.mlp( torch.cat((split_full_tree_embedding, code_token_embedding), 1)  )
            # [bs, hid_size]
            
        else:
            return self.encoder(nl_inputs, attention_mask=nl_inputs.ne(1))[1]
            # [bs, hid_size]
