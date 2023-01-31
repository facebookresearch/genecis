#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --job-name=baselines
#SBATCH --partition=devlab
#SBATCH --gres=gpu:8
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --array=0
#SBATCH --output=/checkpoint/sgvaze/slurm_outputs/myLog-%A_%a.out
#SBATCH --chdir=/private/home/sgvaze/conditional_similarity/baselines
#--------------------------

NGPUS=8
PORT=$RANDOM
echo 'Port:'
echo $PORT

PYTHON='/private/home/sgvaze/miniconda3/envs/condsim/bin/python'
cd '/private/home/sgvaze/genecis/eval/'

# -------------
# BASELINES ON A PARTICULAR DATASET
# -------------
MODEL='RN50x4'
COMBINER_MODE='combiner_original'

# COMBINER_PATHS=("/checkpoint/sgvaze/conditional_similarity/cc3m/tb_logs_v3/09.10.2022_93d0/combiner_best.pt")
# CLIP_PATHS=("/checkpoint/sgvaze/conditional_similarity/cc3m/tb_logs_v3/09.10.2022_93d0/clip_model_best.pt")

COMBINER_PATHS=("/checkpoint/sgvaze/conditional_similarity/cc3m/tb_logs_v3/12.01.2023_12b2/combiner_best.pt")
CLIP_PATHS=("/checkpoint/sgvaze/conditional_similarity/cc3m/tb_logs_v3/12.01.2023_12b2/clip_model_best.pt")

COMBINER_PRETRAIN_PATH=${COMBINER_PATHS[$SLURM_ARRAY_TASK_ID]}
CLIP_PRETRAIN_PATH=${CLIP_PATHS[$SLURM_ARRAY_TASK_ID]}

DATASET='CIRR'
EVAL_MODE='subset'
EVAL_VERSION='v3'
${PYTHON} -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port $PORT evaluate_baselines.py --dataset $DATASET --eval_mode $EVAL_MODE --eval_version $EVAL_VERSION\
                                                                                --model $MODEL --combiner_mode $COMBINER_MODE\
                                                                                --combiner_pretrain_path $COMBINER_PRETRAIN_PATH --clip_pretrain_path $CLIP_PRETRAIN_PATH

DATASET='mit_states'
${PYTHON} -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port $PORT evaluate_baselines.py --dataset $DATASET --eval_mode $EVAL_MODE --eval_version $EVAL_VERSION\
                                                                                --model $MODEL --combiner_mode $COMBINER_MODE\
                                                                                --combiner_pretrain_path $COMBINER_PRETRAIN_PATH --clip_pretrain_path $CLIP_PRETRAIN_PATH

DATASET='focus_attribute'
${PYTHON} -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port $PORT evaluate.py --dataset $DATASET --eval_mode $EVAL_MODE --eval_version $EVAL_VERSION\
                                                                                --model $MODEL --combiner_mode $COMBINER_MODE\
                                                                                --combiner_pretrain_path $COMBINER_PRETRAIN_PATH --clip_pretrain_path $CLIP_PRETRAIN_PATH

DATASET='change_attribute'
${PYTHON} -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port $PORT evaluate.py --dataset $DATASET --eval_mode $EVAL_MODE --eval_version $EVAL_VERSION\
                                                                                --model $MODEL --combiner_mode $COMBINER_MODE\
                                                                                --combiner_pretrain_path $COMBINER_PRETRAIN_PATH --clip_pretrain_path $CLIP_PRETRAIN_PATH

DATASET='focus_object'
${PYTHON} -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port $PORT evaluate.py --dataset $DATASET --eval_mode $EVAL_MODE --eval_version $EVAL_VERSION\
                                                                                --model $MODEL --combiner_mode $COMBINER_MODE\
                                                                                --combiner_pretrain_path $COMBINER_PRETRAIN_PATH --clip_pretrain_path $CLIP_PRETRAIN_PATH

DATASET='change_object'
${PYTHON} -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port $PORT evaluate.py --dataset $DATASET --eval_mode $EVAL_MODE --eval_version $EVAL_VERSION\
                                                                                --model $MODEL --combiner_mode $COMBINER_MODE\
                                                                                --combiner_pretrain_path $COMBINER_PRETRAIN_PATH --clip_pretrain_path $CLIP_PRETRAIN_PATH