## To evaluate on GeneCIS 

The GeneCIS dataset builds upon [Visual Genome](http://visualgenome.org/) and [COCO](https://cocodataset.org/#home). Specifically, we use images from [Visual Genome 1.2](http://visualgenome.org/api/v0/api_home.html) and validation images from the [COCO 2017 split](https://cocodataset.org/#download). 

After downloading, specify the paths to the images in ```config.py```. There should be 108249 in the Visual Genome directory, and 5000 images in the COCO validation directory.

## To train on Conceptual Captions 3M (CC3M)
You will first need to download the image-caption data from [here](https://ai.google.com/research/ConceptualCaptions/download). Specify the directory in ```config.py```, and its contents should look like:
```
├── train_all.npy
├── training/
├── val.npy
└── validation/
```

This repo contains scripts to parse scene graphs from CC3M and to mine training triplets. Alternatively, you can download the [scene graphs](TODO) and [training triplets](TODO) from here. 

## Pre-trained weights

| CLIP Backbone | Training Set | GeneCIS Avg. Recall @ 1 | Backbone | Combiner |
|---------------|--------------|-------------------------|----------|----------|
| RN50x4        | CC3M         | 16.8                    | [download](TODO) | [download](TODO) |
| ViT-B/16      | CC3M         | 17.6                    | [download](TODO) | [download](TODO) |
