# Welcome to Sparse Fusion Mixture-of-Experts for Domain Generalization

### Performance Comparison

$x$-axis is the training iteration time per mini-batch with 160 images (lower is better). 

$y$-axis is the overall accuracy on DomainNet with training-validation model selection criterion (higher is better). 

The **bubble size** and the text floating around demonstrate the run-time memory cost during training (smaller is better)
<p align="center">
    <img src="./assets/comp.png" width="100%" />
</p>

### Diagram of SF-MoE
<p align="center">
    <img src="./assets/teaser.png" width="100%" />
</p>

## Preparation

### Dependencies

```sh
pip install -r requirements.txt
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
	Python: 3.9
	PyTorch: 1.8.0
	Torchvision: 0.8.2
	CUDA: 10.2
	CUDNN: 7603
	NumPy: 1.21.4
	PIL: 7.2.0
```

## Start Training

Train a model:

```sh
python3 -m domainbed.scripts.train\
       --data_dir=./domainbed/data/OfficeHome/\
       --algorithm SFMOE\
       --dataset OfficeHome\
       --test_env 2
```

Launch a sweep:

```sh
python -m domainbed.scripts.sweep launch\
       --data_dir=/my/datasets/path\
       --output_dir=/my/sweep/output/path\
       --command_launcher multi_gpu
```

To view the results of your sweep:

````sh
python -m domainbed.scripts.collect_results\
       --input_dir=/my/sweep/output/path
````

Our Hyper-parameters for each dataset:


|                    | PACS | VLCS | OfficeHome | TerraIncognita | DomainNet |
|--------------------| ------ | ------ | ------------ | ---------------- | ----------- |
| Learningfafsfarate | 3e-5 | 1e-5 | 3e-5       | 3e-5           | 3e-5      |
| Dropout            | 0.0  | 0.5  | 0.1        | 0.0            | 0.1       |
| Weight decay       | 0.0  | 1e-6 | 1e-6       | 1e-4           | 0.0       |

## Experimental Results

### Available model selection criteria

[Model selection criteria](domainbed/model_selection.py) differ in what data is used to choose the best hyper-parameters for a given model:

* `IIDAccuracySelectionMethod`: A random subset from the data of the training domains.
* `LeaveOneOutSelectionMethod`: A random subset from the data of a held-out (not training, not testing) domain.

### Train-val selection strategy
<p align="center">
    <img src="./assets/train-val.png" width="90%" />
</p>

### Leave-one-domain-out selection strategy
<p align="center">
    <img src="./assets/lodo.png" width="90%" />
</p>

### Multi-heads Attention Visualization
<p align="center">
    <img src="./assets/mha.png" width="90%" />
</p>

## License

This source code is released under the MIT license, included [here](LICENSE).
