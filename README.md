# Welcome to Generalizable Mixture-of-Experts for Domain Generalization

🔥 Our paper [Sparse Mixture-of-Experts are Domain Generalizable Learners](https://openreview.net/forum?id=RecZ9nB9Q4) has officially been accepted as ICLR 2023 for Oral presentation. 

🔥 GMoE-S/16 model currently [ranks top place](https://paperswithcode.com/sota/domain-generalization-on-domainnet) among multiple DG datasets without extra pre-training data. (Our GMoE-S/16 is initilized from [DeiT-S/16](https://github.com/facebookresearch/deit/blob/main/README_deit.md), which was only pretrained on ImageNet-1K 2012)

Wondering why GMoEs have astonishing performance? 🤯 Let's investigate the generalization ability of model architecture itself and see the great potentials of Sparse Mixture-of-Experts (MoE) architecture.

### Preparation

```sh
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

python3 -m pip uninstall tutel -y
python3 -m pip install --user --upgrade git+https://github.com/microsoft/tutel@main

pip3 install -r requirements.txt
```

### Datasets

```sh
python3 -m domainbed.scripts.download \
       --data_dir=./domainbed/data
```

### Environments

Environment details used in paper for the main experiments on Nvidia V100 GPU.

```shell
Environment:
	Python: 3.9.12
	PyTorch: 1.12.0+cu116
	Torchvision: 0.13.0+cu116
	CUDA: 11.6
	CUDNN: 8302
	NumPy: 1.19.5
	PIL: 9.2.0
```

## Start Training

Train a model:

```sh
python3 -m domainbed.scripts.train\
       --data_dir=./domainbed/data/OfficeHome/\
       --algorithm GMOE\
       --dataset OfficeHome\
       --test_env 2
```

## Hyper-params

We put hparams for each dataset into
```sh
./domainbed/hparams_registry.py
```

Basically, you just need to choose `--algorithm` and `--dataset`. The optimal hparams will be loaded accordingly. 

## License

This source code is released under the MIT license, included [here](LICENSE).

## Acknowledgement

The MoE module is built on [Tutel MoE](https://github.com/microsoft/tutel).
