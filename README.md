# long-summarization
Pytorch implementation of NAACL 2018 ["A Discourse-Aware Attention Model for Abstractive Summarization of Long Documents"](https://arxiv.org/abs/1804.05685) [(code)](https://github.com/acohan/long-summarization)

### Data

Two datasets of long documents are provided by original authors.

ArXiv dataset: [Download](https://drive.google.com/file/d/1K2kDBTNXS2ikx9xKmi2Fy0Wsc5u_Lls0/view?usp=sharing)  
PubMed dataset: [Download](https://drive.google.com/file/d/1Sa3kip8IE0J1SkMivlgOwq1jBgOnzeny/view?usp=sharing)

We also provide small try_out.zip file for the test purpose.

### Requirements
- Python 3.7
- Pytorch 1.0.1
- Pyrouge

### Run training
`src/main.py -data pubmed -save_dir SAVE_DIR`


### Run decoding
`src/decode.py -data pubmed -mode decode -train_from MODEL_PATH`