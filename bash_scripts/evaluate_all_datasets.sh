#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --job-name=eval
#SBATCH --partition=devlab
#SBATCH --gres=gpu:8
#SBATCH --mem=100G
#SBATCH --cpus-per-task=10
#SBATCH --array=0-2
#SBATCH --output=/checkpoint/sgvaze/slurm_outputs/myLog-%A_%a.out
#SBATCH --chdir=/private/home/sgvaze/genecis/
#--------------------------

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
MODEL='ViT-B/16'
# COMBINER_MODE='combiner_original'
COMBINER_MODES=('image_only' 'text_only' 'image_plus_text')
COMBINER_MODE=${COMBINER_MODES[$SLURM_ARRAY_TASK_ID]}

# -------------
# DEFINE PRETRAINED PATHS
# -------------
# COMBINER_PRETRAIN_PATH="/checkpoint/sgvaze/conditional_similarity/cc3m/tb_logs_v3/09.10.2022_93d0/combiner_best.pt"
# CLIP_PRETRAIN_PATH="/checkpoint/sgvaze/conditional_similarity/cc3m/tb_logs_v3/09.10.2022_93d0/clip_model_best.pt"
# COMBINER_PRETRAIN_PATH="/checkpoint/sgvaze/conditional_similarity/cc3m/tb_logs_v3/10.10.2022_2d38/combiner_best.pt"
# CLIP_PRETRAIN_PATH="/checkpoint/sgvaze/conditional_similarity/cc3m/tb_logs_v3/10.10.2022_2d38/clip_model_best.pt"
COMBINER_PRETRAIN_PATH="None"
CLIP_PRETRAIN_PATH="None"

# -------------
# INIT
# -------------
export PYTHONPATH=$PROJECT_ROOT
cd $PROJECT_ROOT
cd eval/
echo "Port:$PORT"

DATASET='CIRR'
${PYTHON} -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port $PORT evaluate.py --dataset $DATASET\
                                                                                --model $MODEL --combiner_mode $COMBINER_MODE\
                                                                                --combiner_pretrain_path $COMBINER_PRETRAIN_PATH --clip_pretrain_path $CLIP_PRETRAIN_PATH

DATASET='mit_states'
${PYTHON} -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port $PORT evaluate.py --dataset $DATASET\
                                                                                --model $MODEL --combiner_mode $COMBINER_MODE\
                                                                                --combiner_pretrain_path $COMBINER_PRETRAIN_PATH --clip_pretrain_path $CLIP_PRETRAIN_PATH

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