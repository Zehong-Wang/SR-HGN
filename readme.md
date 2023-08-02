# SR-HGN

This repo is for source code of Expert Systems with Applications paper "**SR-HGN: Semantic-and Relation-Aware Heterogeneous Graph Neural Network**". [paper](https://www.sciencedirect.com/science/article/abs/pii/S0957417423004840)

## Environment Settings

* python==3.8.0
* scipy==1.6.2
* torch==1.11.0
* scikit-learn==1.0.2
* torch_geometric==2.0.4
* dgl==0.6.1

## Dataset

We utilize three benchmark datasets in the paper to perform node classification and node clustering. We provide ACM, DBLP, and IMDB in [GoogleDrive](https://drive.google.com/drive/folders/1KvqXpi4NDSiTkPe_Bsgt9AqnLMx2tbka?usp=sharing). 

* ACM
* DBLP
* IMDB

You can create the "data" folder in the root directory, then put the datasets in. Like "/SRHGN/data/acm/...". 
 
## How to run

For example, if you want to run SR-HGN on ACM dataset, execute

```
python main.py --dataset acm
```

## Citation
```
@article{wang2023sr,
  title={SR-HGN: Semantic-and Relation-Aware Heterogeneous Graph Neural Network},
  author={Wang, Zehong and Yu, Donghua and Li, Qi and Shen, Shigen and Yao, Shuang},
  journal={Expert Systems with Applications},
  volume={224},
  pages={119982},
  year={2023},
  publisher={Elsevier}
}
```

## Contact

If you have any questions, don't hesitate to contact me (zwang43@nd.edu, zehongwang0414@gmail.com)! 