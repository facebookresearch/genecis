# -------------
# DEFINE EVAL PARAMS
# -------------
PYTHON='/private/home/sgvaze/miniconda3/envs/genecis/bin/python'
PROJECT_ROOT='/private/home/sgvaze/genecis/'
NGPUS=2
PORT=$RANDOM

# -------------
# DEFINE MODEL SPECS
# -------------
MODEL='RN50x4'          # Set to one of the CLIP models
COMBINER_MODE='combiner_original'           # Either learned combiner head or one of ('image_only' 'text_only' 'image_plus_text')

# -------------
# DEFINE PRETRAINED PATHS
# -------------
COMBINER_PRETRAIN_PATH="/checkpoint/sgvaze/conditional_similarity/cc3m/tb_logs_v3/09.10.2022_93d0/combiner_best.pt"               # Set to path of model to evaluate (combiner head)  (set to 'None' if using image_only etc.)
CLIP_PRETRAIN_PATH="/checkpoint/sgvaze/conditional_similarity/cc3m/tb_logs_v3/09.10.2022_93d0/clip_model_best.pt"                   # Set to path of model to evaluate (backbone)  (set to 'None' to use CLIP pre-trained model, if using image_only etc.)

# -------------
# INIT
# -------------
export PYTHONPATH=$PROJECT_ROOT
cd $PROJECT_ROOT
cd eval/
echo "Port:$PORT"

# =================== GeneCIS EVAL ===================

DATASET='focus_attribute'
${PYTHON} -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port $PORT evaluate.py --dataset $DATASET\
                                                                                --model $MODEL --combiner_mode $COMBINER_MODE\
                                                                                --combiner_pretrain_path $COMBINER_PRETRAIN_PATH --clip_pretrain_path $CLIP_PRETRAIN_PATH

DATASET='change_attribute'
${PYTHON} -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port $PORT evaluate.py --dataset $DATASET\
                                                                                --model $MODEL --combiner_mode $COMBINER_MODE\
                                                                                --combiner_pretrain_path $COMBINER_PRETRAIN_PATH --clip_pretrain_path $CLIP_PRETRAIN_PATH

DATASET='focus_object'
${PYTHON} -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port $PORT evaluate.py --dataset $DATASET\
                                                                                --model $MODEL --combiner_mode $COMBINER_MODE\
                                                                                --combiner_pretrain_path $COMBINER_PRETRAIN_PATH --clip_pretrain_path $CLIP_PRETRAIN_PATH

DATASET='change_object'
${PYTHON} -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port $PORT evaluate.py --dataset $DATASET\
                                                                                --model $MODEL --combiner_mode $COMBINER_MODE\
                                                                                --combiner_pretrain_path $COMBINER_PRETRAIN_PATH --clip_pretrain_path $CLIP_PRETRAIN_PATH


# =================== CIR DATASETS ===================

DATASET='CIRR'
${PYTHON} -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port $PORT evaluate.py --dataset $DATASET\
                                                                                --model $MODEL --combiner_mode $COMBINER_MODE\
                                                                                --combiner_pretrain_path $COMBINER_PRETRAIN_PATH --clip_pretrain_path $CLIP_PRETRAIN_PATH

DATASET='mit_states'
${PYTHON} -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port $PORT evaluate.py --dataset $DATASET\
                                                                                --model $MODEL --combiner_mode $COMBINER_MODE\
                                                                                --combiner_pretrain_path $COMBINER_PRETRAIN_PATH --clip_pretrain_path $CLIP_PRETRAIN_PATH