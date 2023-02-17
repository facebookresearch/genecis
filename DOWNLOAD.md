# Download instructions

Note: After downloading all files and directories, their paths should be specified in ```config.py```.

## To evaluate on GeneCIS 

The GeneCIS dataset builds upon [Visual Genome](http://visualgenome.org/) and [COCO](https://cocodataset.org/#home). Specifically, we use images from [Visual Genome 1.2](http://visualgenome.org/api/v0/api_home.html) and validation images from the [COCO 2017 split](https://cocodataset.org/#download). 

After downloading, specify the paths to the images in ```config.py```. There should be 108249 in the Visual Genome directory, and 5000 images in the COCO validation directory.

## Pre-trained weights

We release weights both from our main evaluations (Table 2), which is initialized with CLIP RN50x4 backbone, and for a stronger model (Table 6, Appendix) which is initialized with CLIP ViT-B/16. 

| CLIP Backbone | Training Set | GeneCIS Avg. Recall @ 1 | Backbone | Combiner |
|---------------|--------------|-------------------------|----------|----------|
| RN50x4        | CC3M         | 16.8                    | [download](https://dl.fbaipublicfiles.com/genecis/rn50x4_backbone.pt) | [download](https://dl.fbaipublicfiles.com/genecis/rn50x4_combiner_head.pt) |
| ViT-B/16      | CC3M         | 17.6                    | [download](https://dl.fbaipublicfiles.com/genecis/vitb16_backbone.pt) | [download](https://dl.fbaipublicfiles.com/genecis/vitb16_combiner_head.pt) |

## To train on Conceptual Captions 3M (CC3M)
1. You will first need to download the image-caption data from [here](https://ai.google.com/research/ConceptualCaptions/download). The directory contents should look like:
```
├── train_all.npy
├── training/
├── val.npy
└── validation/
```

2. This repo contains scripts to parse scene graphs from CC3M and to mine training triplets. To do so, you will need to download the database of noun visual concreteness from [here](http://crr.ugent.be/archives/1330).
Alternatively, you can download the [scene graphs](https://dl.fbaipublicfiles.com/genecis/cc3m_scene_graphs.pt) and [training triplets](https://dl.fbaipublicfiles.com/genecis/cc3m_training_triplets_1.6M.pt) from here. 

## For additional evaluations

1. We use CIRR as a validation set. If you wish to do this, download following the official instructions [here](https://github.com/Cuberick-Orion/CIRR).
Note that this will also require downloading raw images from NLVR2, for which access can be requested [here](https://lil.nlp.cornell.edu/nlvr/).
The final dataset structure should include ```cirr``` annotations and raw image data as:
```
├── cirr
├── dev
├── test1
└── train
```

2. We also evaluate on MIT-States, this dataset can be downloaded [here](http://web.mit.edu/phillipi/Public/states_and_transformations/index.html).
The dataset structure should be: 
```
├── README.txt
├── __MACOSX
├── adj_ants.csv
└── images
```