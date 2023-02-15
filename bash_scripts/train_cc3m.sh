# Project paths and init
PYTHON='/private/home/sgvaze/miniconda3/envs/genecis/bin/python'
PROJECT_ROOT='/private/home/sgvaze/genecis/'

export PYTHONPATH=$PROJECT_ROOT
cd $PROJECT_ROOT
cd train/

# Model params
LAMDA=100
COMBINER_MODE='combiner_original'
MODEL='RN50x4'
OPTIMIZER='adamw'
FINETUNE_MODE='None'
WEIGHT_DECAY=0.0
LR=1e-5
EPOCHS=100
BATCH_SIZE_PER_GPU=128
SCHEDULER='cosine'

# Loss
LOSS='base_contrastive'

# Data
TRAIN_DATASET='coco'
MIN_IMAGES_WITH_SUBJECT=5
NUM_SAMPLES_PER_EPOCH=60000
NUM_WORKERS=10
DETERM_SAMPLES_KEY='ConditionalDistractor'
CONCRETENESS_THRESH=-1.0
PROMPT_PREPEND="None"
VAL_START_IDX=0

# Misc
SEED=0

${PYTHON} -m torch.distributed.launch --nproc_per_node=2 train_cc3m.py --num_workers $NUM_WORKERS\
                                     --lamda $LAMDA --num_samples_per_epoch $NUM_SAMPLES_PER_EPOCH\
                                     --min_images_with_subject $MIN_IMAGES_WITH_SUBJECT  --model $MODEL --scheduler $SCHEDULER --combiner_mode $COMBINER_MODE --epochs $EPOCHS\
                                     --lr $LR --optimizer $OPTIMIZER --batch_size_per_gpu $BATCH_SIZE_PER_GPU --finetune_mode $FINETUNE_MODE --weight_decay $WEIGHT_DECAY\
                                     --seed $SEED --prompt_prepend $PROMPT_PREPEND --deterministic_samples_key $DETERM_SAMPLES_KEY --concreteness_threshold $CONCRETENESS_THRESH\
                                     --train_dataset $TRAIN_DATASET --val_start_idx $VAL_START_IDX