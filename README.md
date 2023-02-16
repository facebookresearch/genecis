# GeneCIS: A Benchmark for General Conditional Image Similarity

GeneCIS (pronounced 'genesis') is a zero-shot evaluation benchmark for measuring vision models' ability to adapt to various notions of visual 'similarity' given a text prompt. This repo contains evaluation boiler plate evaluation code for the benchmark, as well as code to train a model on Conceptual Caption 3M to tackle it.

Find more details in our [paper](TODO).

<p align="center"> <img src='assets/genecis_examples.png' align="center" > </p>

## Contents
[:computer: 1. Installation](#install)

[:arrow_down: 2. Downloads](#downloads)

[:notebook_with_decorative_cover: 3. Pre-processing](#preproc)

[:train: 4. Training](#training)

[:clipboard: 5. Citation](#cite)

## <a name="install"/> :computer: Installation

```bash
# Create virtual env 
conda create -n genecis python=3.9.12
conda activate genecis
conda install --yes -c pytorch pytorch=1.12.0 torchvision=0.13.0 cudatoolkit=11.3.1     # Change toolkit version if necessary
pip install git+https://github.com/openai/CLIP.git

# Under your working directory
git clone git@github.com:facebookresearch/genecis.git
cd genecis
pip install -r requirements.txt
python -m spacy download en         # Required for scene graph parsing
```
## <a name="downloads"/> :arrow_down: Downloads

Download instructions can be found [here](/DOWNLOAD.md). You will need to download the raw images from the COCO 2017 validation split and VisualGenome1.2 to evaluate on GeneCIS.
You can also find instructions to download: parsed CC3M scene graphs; 1.6M mined triplets from CC3M; and pre-trained model weights; additional evaluation datasets.

## <a name="preproc"/> :notebook_with_decorative_cover: Pre-processing

We train models for GeneCIS using image-caption data from Conceptual Captions 3M. 
Before training the models, we need to [parse scene graphs](train/extract_scene_graphs.py) and [mine training triplets](train/create_deterministic_samples.py) from the raw data. 
The scene graphs and triplets can be downloaded following instructions [here](/DOWNLOAD.md), or else processed directly. 
To do this, set hyper-parameters in ```config.py``` (or leave in the default setting), and look at the example commands in ```bash_scripts/mine_training_samples.sh```.

Especially, mining the training triplets can take a long time if done in series, but running the following command in parallel with SHARD_INDEX ranging from 0 to ```config.NUM_SHARDS - 1``` will speed up the process:
```
python -m train.create_deterministic_samples --shard_index SHARD_INDEX
```

## <a name="training"/> :train: Training

Training the models with default hyper-parameters requires training on 16GPUs. To do this with multi-node training, edit paths and run: 

```
bash_scripts/submitit_train_cc3m.sh
```

Otherwise, for single node training, an example command is given in:

```
bash_scripts/train_cc3m.sh
```

## <a name="cite"/> :clipboard: Citation

If you use this code in your research, please consider citing our paper:
```
@article{vaze2023gen,
        title={GeneCIS: A Benchmark for General Conditional Image Similarity},
        author={Sagar Vaze and Nicolas Carion and Ishan Misra},
        journal={arXiv preprint},
        year={2023}
        }
```