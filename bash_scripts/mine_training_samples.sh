PYTHON='/private/home/sgvaze/miniconda3/envs/genecis/bin/python'

# Extract scene graphs from CC3M captions
${PYTHON} -m train.extract_scene_graphs         # This should take ~5 hours

# All triplets can be created in series, which will take around 70 hours
${PYTHON} -m train.create_deterministic_samples     

# Alternatively, we can generate the samples in shards and concatenate them afterwards
${PYTHON} -m train.create_deterministic_samples --shard_index 0
${PYTHON} -m train.combine_deterministic_samples 