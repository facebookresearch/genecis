# Project paths and init
PYTHON='/private/home/sgvaze/miniconda3/envs/genecis/bin/python'
PROJECT_ROOT='/private/home/sgvaze/genecis/'

export PYTHONPATH=$PROJECT_ROOT
cd $PROJECT_ROOT

# ---------------
# Extract scene graphs from CC3M captions
# ---------------
${PYTHON} -m train.extract_scene_graphs         # This should take ~5 hours

# ---------------
# Mine deterministic training samples
# ---------------
# Method 1: All triplets can be created in series, which will take around 70 hours
${PYTHON} -m train.create_deterministic_samples     

# Method 2: Alternatively, we can generate the samples in shards and concatenate them afterwards
${PYTHON} -m train.create_deterministic_samples --shard_index 0     # Run in parallel with shard index from 0-399. This takes ~10mins 
${PYTHON} -m train.combine_deterministic_samples 