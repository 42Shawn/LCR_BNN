# LCR-BNN: Lipschitz Continuity Retained Binary Neural Network
The code for the Lipschitz Continuity Retained Binary Neural Network, which has been accepted to ECCV 2022.
## Quick Start
First, download our repo:
```bash
git clone https://github.com/42Shawn/LCR_BNN.git
cd LCR_BNN
```
Then, run our repo:
```bash
python main.py --save='v0' --data_path='path-to-dataset' --gpus='gpu-id' --alpha=3.2
```
Note that the alpha can be change to conduct ablation studies, and alpha=0 is equal to RBNN itself.

# Reference
If you find our code useful for your research, please cite our paper.
```
@inproceedings{
shang2022lcr,
title={Lipschitz Continuity Retained Binary Neural Network},
author={Yuzhang Shang and Dan Xu and Bin Duan and Ziliang Zong and Liqiang Nie and Yan Yan},
booktitle={ECCV},
year={2022}
}
```

**Related Work**    
Our repo is modified based on the Pytorch implementation of Rotated Binary Neural Network (RBNN, NeurIPS 2020). Thanks to the authors for releasing their codebase!
