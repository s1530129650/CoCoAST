
# Code Search

## Environment

Create an environment and activate it:

```
conda create -n  CAST python=3.6 ipykernel -y
conda activate CAST
conda install pydot -y
conda install pyzmq -y
pip install git+https://github.com/casics/spiral.git
pip install pandas==1.0.5 gensim==3.5.0 scikit-learn==0.19.1 pycparser==2.18 javalang==0.11.0 tqdm networkx==2.3 nltk==3.6 psutil gin-config prettytable
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
pip install tqdm prettytable  transformers==4.12.5 gdown more-itertools tensorboardX  sklearn
pip install tree_sitter seaborn==0.11.2 fast-histogram
```

Install Java 1.8

Install pytorch according to your environment, see https://pytorch.org/ 
you can use `nvcc --version` to check the cuda version. 

##  Data Preparation 

### Downloading and preprocessing codesearchnet java dataset
```
cd dataset
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/java.zip
unzip java.zip
rm *.zip
python preprocess.py
```
it will generate `train.jsonl`,`valid.jsonl`,`test.jsonl` and `codebase.jsonl`

### Obtaining split asts

```
bash get_split_ast.sh 
```

``` shell
% getting ast vocab
python get_flatten_ast_and_ast_vocab.py
```
## CAST Model
run.py or run_astnn.py
### Training
```
bash run.sh
```


##  Codebert/graphcodbert moldel

run_transformer.py

   
|model_type|description|||
| :--------- | :------: | :----: | :----: |
| transformer + use_pre_trained|  Roberta,Codebert, graphcodebert, codebert-mlm |||
| Roberta_from_scratch   | train Roberta from scratch|||
| X_RM_POS | remove position encoder|||