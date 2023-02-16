# -----------------
# Dataset paths
# -----------------
# GeneCIS splits
genecis_root = '/private/home/sgvaze/genecis/genecis'                       # Path to GeneCIS templates (/project/dir/genecis)

# GeneCIS base datasets
visual_genome_images = '/datasets01/VisualGenome1.2/061517/VG_100K_all'     # Version 1.2, all images
coco_root = '/datasets01/COCO/022719/val2017'                               # 2017 validation images

# Conceptual Captions 3M
cc3m_root = '/large_experiments/cmd/cc'                                     # All images and captions

# For additional evaluations on MIT States and CIRR
mit_states_root = '/checkpoint/sgvaze/datasets/mit_states'
cirr_root = '/checkpoint/sgvaze/datasets/cirr_data'                         # Required as a validation set

# -----------------
# Metadata paths
# -----------------
cc3m_tsg_path = '/private/home/sgvaze/conditional_similarity/cc_tsg_3m.pt'                                                                                                              # Where to save/look for parsed scene graphs from CC3M
cc3m_deterministic_root_path = '/checkpoint/sgvaze/conditional_similarity/cc3m_deterministic_samples/CCConditionalDistractor_1.6E+06_4.8_default_concreteness.pt'           # Where to save/look for mined training triplets from CC3M
noun_concreteness_score_path = '/checkpoint/sgvaze/conditional_similarity/misc/Concreteness_ratings_Brysbaert_et_al_BRM.txt'                                                # Database of noun concreteness

assert cc3m_deterministic_root_path[-3:] == '.pt', 'cc3m_deterministic_root_path should be a path to a pytorch file'

# -----------------
# Hyper-params and logging
# -----------------
log_dir = '/checkpoint/sgvaze/conditional_similarity/cc3m/tb_logs_v3'      # Tensorboard logging
num_deterministic_samples = 1.6e6                                           # How many triplets to mine from CC3M
cc3m_concreteness_threshold = 4.8                                           # Concreteness threshold for filtering mined triplets
NUM_SHARDS = 400                                                            # How many shards to generate triplets in
