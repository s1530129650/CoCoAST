conda create -n  CAST python=3.6 ipykernel -y
conda activate CAST
source ~/miniconda3/etc/profile.d/conda.sh
source ~/anaconda3/etc/profile.d/conda.sh
conda install pydot -y
conda install pyzmq -y
pip install git+https://github.com/casics/spiral.git
pip install pandas==1.0.5 gensim==3.5.0 scikit-learn==0.19.1 pycparser==2.18 javalang==0.11.0 tqdm networkx==2.3 nltk==3.6 psutil gin-config prettytable
pip install torch==1.10 tqdm prettytable  transformers==4.12.5 gdown more-itertools tensorboardX  sklearn
pip install tree_sitter seaborn==0.11.2 fast-histogram