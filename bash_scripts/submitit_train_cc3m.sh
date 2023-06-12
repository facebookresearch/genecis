# Project paths and init
PYTHON='/private/home/sgvaze/miniconda3/envs/genecis/bin/python'
PROJECT_ROOT='/private/home/sgvaze/genecis/'

export PYTHONPATH=$PROJECT_ROOT
cd $PROJECT_ROOT

# Submitit params
PARTITION='devlab'
NODES=2
GPUS=8

# Model params
LAMDA=100
COMBINER_MODE='combiner_original'
CLIP_BACKBONE='RN50x4'

# Data
CONCRETENESS_THRESH=4.8
VAL_START_IDX=0

# Optim
OPTIMIZER='adamw'
FINETUNE_MODE='image_plus_text_whole'
WEIGHT_DECAY=0.0
LR=1e-6
BATCH_SIZE_PER_GPU=16
SCHEDULER='cosine'
NUM_SAMPLES_PER_EPOCH=1600000
EPOCHS=6

# Misc
SEED=0
NUM_WORKERS=10
NGPUS=2

${PYTHON} run_cc3m_train.py --partition $PARTITION --ngpus $GPUS --nodes $NODES --num_workers $NUM_WORKERS\
                                     --lamda $LAMDA --num_samples_per_epoch $NUM_SAMPLES_PER_EPOCH\
                                     --backbone $CLIP_BACKBONE --scheduler $SCHEDULER --combiner_mode $COMBINER_MODE --epochs $EPOCHS\
                                     --lr $LR --optimizer $OPTIMIZER --batch_size_per_gpu $BATCH_SIZE_PER_GPU --finetune_mode $FINETUNE_MODE --weight_decay $WEIGHT_DECAY\
                                     --seed $SEED --val_start_idx $VAL_START_IDX