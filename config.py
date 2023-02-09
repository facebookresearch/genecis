# Dataset roots
# GeneCIS base datasets
visual_genome_images = '/datasets01/VisualGenome1.2/061517/VG_100K_all'     # Version 1.2, all images
coco_root = '/datasets01/COCO/022719/val2017'                               # Validation split

# Conceptual Captions 3M
cc3m_root = '/large_experiments/cmd/cc'                                     # All images and captions

# GeneCIS splits
genecis_root = '/private/home/sgvaze/genecis/genecis'

# For additional evaluations on MIT States and CIRR
mit_states_root = '/checkpoint/sgvaze/datasets/mit_states'
cirr_root = '/checkpoint/sgvaze/datasets/cirr_data'

# Experiment and meta data paths
cc3m_tsg_path = '/checkpoint/sgvaze/genecis_test/cc_tsg_3m.pt'
cc3m_tsg_path = '/private/home/sgvaze/conditional_similarity/cc_tsg_3m.pt'
cc3m_deterministic_root_path = '/checkpoint/sgvaze/genecis_test/mined_triplets_1.6E6_4.8_thresh.pt'
noun_concreteness_score_path = '/checkpoint/sgvaze/conditional_similarity/misc/Concreteness_ratings_Brysbaert_et_al_BRM.txt'
log_dir = '/checkpoint/sgvaze/conditional_similarity/cc3m/tb_logs_v3'

assert cc3m_deterministic_root_path[-3:] == '.pt', 'cc3m_deterministic_root_path should be a path to a pytorch file'

# Params for CC3M triplets
num_deterministic_samples = 1.6e6
cc3m_concreteness_threshold = 4.8
cc3m_min_images_with_subject = 5
NUM_SHARDS = 400                # How many parallel shards to generate deterministic triplets with