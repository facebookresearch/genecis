PYTHON='/private/home/sgvaze/miniconda3/envs/condsim/bin/python'

cd /private/home/sgvaze/conditional_similarity/

# Submitit params
PARTITION='devlab'
NODES=2
GPUS=8
BATCH_SIZE_PER_GPU=16
FINETUNE_MODE='image_plus_text_whole'
LR=1e-6

# Model params
NORM_FEATS_BEFORE_COMBINE='False'
LAMDA=100
COMBINER_MODE='combiner_original'
MODEL='RN50x4'
OPTIMIZER='adamw'
WEIGHT_DECAY=0.0
EPOCHS=100
SCHEDULER='cosine'

# Data
MIN_IMAGES_WITH_SUBJECT=5
NUM_SAMPLES_PER_EPOCH=1600000
NUM_WORKERS=10
CONCRETENESS_THRESH=4.8
PROMPT_PREPEND="None"
VAL_START_IDX=0
TSG_PATH_KEY='default'

# Misc
SEED=0

${PYTHON} run_cc3m_train.py --partition $PARTITION --ngpus $GPUS --nodes $NODES --num_workers $NUM_WORKERS\
                                     --lamda $LAMDA --num_samples_per_epoch $NUM_SAMPLES_PER_EPOCH\
                                     --min_images_with_subject $MIN_IMAGES_WITH_SUBJECT  --model $MODEL --scheduler $SCHEDULER --combiner_mode $COMBINER_MODE --epochs $EPOCHS\
                                     --lr $LR --optimizer $OPTIMIZER --batch_size_per_gpu $BATCH_SIZE_PER_GPU --finetune_mode $FINETUNE_MODE --weight_decay $WEIGHT_DECAY\
                                     --seed $SEED --deterministic_samples_key $DETERM_SAMPLES_KEY --prompt_prepend $PROMPT_PREPEND --concreteness_threshold $CONCRETENESS_THRESH\
                                     --train_dataset $TRAIN_DATASET --norm_feats_before_combining $NORM_FEATS_BEFORE_COMBINE --val_start_idx $VAL_START_IDX --tsg_path_key $TSG_PATH_KEY