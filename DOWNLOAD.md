## GeneCIS base datasets

The GeneCIS dataset builds upon [Visual Genome](http://visualgenome.org/) and [COCO](https://cocodataset.org/#home). Specifically, we use images from [Visual Genome 1.2](http://visualgenome.org/api/v0/api_home.html) and validation images from the [COCO 2018 Panoptic Segmentation](https://cocodataset.org/#panoptic-2018) challenge. 

After downloading, specify the paths to the images in ```config.py```. There should be 108249 in the Visual Genome directory, and 5000 images in the COCO validation directory.

## GeneCIS templates

Download the GeneCIS templates from [here](TODO) and place them inside ```./genecis/```. 

## Mined triplets
This repo contains scripts to parse scene graphs from CC3M and to mine training triplets. Alternatively, you can download the [scene graphs](TODO) and [training triplets](TODO) from here. 

## Pre-trained weights
Weights for the model from the paper can be downloaded [here](TODO).
