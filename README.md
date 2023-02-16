# GeneCIS: A Benchmark for General Conditional Image Similarity

GeneCIS (pronounced 'genesis') is a zero-shot evaluation benchmark for measuring vision models' ability to adapt to various notions of 'similarity' given a text prompt. This repo contains evaluation boiler plate evaluation code for the benchmark, as well as code to train a model on Conceptual Caption 3M to tackle it.

<p align="center"> <img src='assets/genecis_examples.png' align="center" > </p>

## Contents
[:computer: 1. Installation](#install)

[:arrow_down: 2. Downloads](#downloads)

[:clipboard: 4. Citation](#cite)

## <a name="install"/> :computer: Installation

```bash
# Create virtual env 
conda create -n genecis python=3.9.12
conda activate genecis
conda install --yes -c pytorch pytorch=1.12.0 torchvision=0.13.0 cudatoolkit=11.3.1
pip install git+https://github.com/openai/CLIP.git

# Under your working directory
git clone git@github.com:facebookresearch/genecis.git
cd genecis
pip install -r requirements.txt
python -m spacy download en         # Required for scene graph parsing
```
## <a name="downloads"/> :arrow_down: Downloads

Download instructions can be found [here](/DOWNLOAD.md). You will need to download the raw images from the COCO 2017 validation split and VisualGenome1.2 to evaluate on GeneCIS.
You can also find instructions to download: parsed CC3M scene graphs; 1.6M mined triplets from CC3M; and pre-trained model weights.

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