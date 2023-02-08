<h1> minREV </h1>

*Inspired by [minGPT](https://github.com/karpathy/minGPT)* 


A PyTorch reimplementation of [Reversible Vision Transformer](https://openaccess.thecvf.com/content/CVPR2022/papers/Mangalam_Reversible_Vision_Transformers_CVPR_2022_paper.pdf) architecture that is prefers simplicity over ~~tricks~~, hackability over ~~tedious organization~~, and interpretability over ~~generality~~. 

It is meant to serve as an educational guide for newcomers that are not familiar with the reversible backpropagation algorithm and reversible vision transformer. 

The entire Reversible Vision Transformer is implemented from scratch in under **<300 lines** of pytorch code, including the memory-efficient reversible backpropagation algorithm (**<100 lines**). Even the driver code is < 150 lines. The repo supports both memory-efficient training and testing on CIFAR-10.  

💥 The CVPR 2021 [oral talk](https://www.youtube.com/watch?v=AWu-f71C4Nk) for a [5-minute introduction](https://www.youtube.com/watch?v=AWu-f71C4Nk) to RevViT. 

💥 A gentle and in-depth [15 minute introduction](https://youtu.be/X_xyt26tkRY?t=3350) to RevViT.  


<h2> Setting Up </h2>

Simple! 🌟

(if using conda for env, otherwise use pip)
```
conda create -n revvit python=3.8
conda activate revvit
conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia
```

<h2> Code Organization </h2>

The code organization is also minimal 💫:

- `rev.py` defines the reversible vision model that supports: 
    -  The vanilla backpropagation 
    -  The memory-efficient reversible backpropagation    
- `main.py` that has the driver code for training on CIFAR-10.

<h2> Running CIFAR-10 </h2>

`python main.py --lr 1e-3 --bs 128 --embed_dim 128 --depth 6 --n_head 8 --epochs 100`

This will achieve `80%+` validation accuracy on CIFAR-10 from scratch training! 

💯 Here are the [Training/Validation Logs](https://api.wandb.ai/links/action_anticipation/d0hqnv67) 

👁️ **Note**: The relatively low accuracy is due to difficulty in training vision transformer (reversible or vanilla) from scratch on small datasets like CIFAR-10. Also likely is that6 a much higher accuracy can be achieved with the same code, using a better [chosen model design and optimization paramaters](https://github.com/tysam-code/hlb-CIFAR10). The authors have done no tuning since this repository is meant for understanding code, not pushing performance. 

<h2> Running ImageNet, Kinetics-400 and more </h2>

For more usecases such as reproducing numbers from [original paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Mangalam_Reversible_Vision_Transformers_CVPR_2022_paper.pdf), see the [full code in PySlowFast](https://github.com/facebookresearch/SlowFast) that supports 

- ImageNet 
- Kinetics-400/600/700 
- RevViT, all sizes with configs 
- RevMViT, all sizes with configs 

to state-of-the-art accuracies. 

