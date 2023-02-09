import os
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from datasets.cirr_datasets import CIRRImageDataset, CIRRValGlobalDataset
from datasets.vaw_dataset import VAWValSubset
from datasets.mit_states_dataset import MITStatesImageOnly, MITStatesGlobalTestSet 
from datasets.coco_dataset import COCOValSubset

from eval.eval_functions import validate, validate_global
from models.combiner_model import Combiner

import clip
import torch

import argparse
import torch.backends.cudnn as cudnn
from utils.dist_utils import fix_random_seeds, init_distributed_mode, get_rank
from utils.gen_utils import bool_flag, strip_state_dict, none_flag
from functools import partial

from utils.model_utils import FeatureComb
from utils.dist_utils import CLIPDistDataParallel

from config import genecis_root

def get_args_parser():

    parser = argparse.ArgumentParser('Eval', add_help=False)

    parser.add_argument('--model', default='RN50x4', type=str, help='Which CLIP model we are using as backbone')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--batch_size_per_gpu', default=8, type=int)

    # Dist params
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")

    # Model params
    parser.add_argument('--eval_mode', default='global', type=str)        # {image/text/combiner}_{subset/global} or image_text_baseline_global
    parser.add_argument('--combiner_mode', default='text_only', type=str)
    parser.add_argument('--feature_comb_average', default=0.5, type=float)

    # Pretrain paths
    parser.add_argument('--clip_pretrain_path', default=None, type=none_flag)
    parser.add_argument('--combiner_pretrain_path', default=None, type=none_flag)

    # Dataset params
    parser.add_argument('--dataset', default='CIRR', type=str, help='Eval dataset')
    parser.add_argument('--use_manual_annots', default=False, type=bool_flag, help='Whether to filter test samples based on manual annotations...')
    parser.add_argument('--use_complete_text_query', default=False, type=bool_flag, help='Only relevant for MIT States')
    parser.add_argument('--eval_version', default='v3', type=str, help='Only valid for VAW and COCO')
    parser.add_argument('--dilation', default=0.7, type=float, help='Only valid for VAW')
    parser.add_argument('--pad_crop', default=True, type=bool_flag, help='Only valid for VAW')

    # Save params
    parser.add_argument('--pred_save_name', default=None, type=none_flag, help='Where to save predictions, dont save by default')

    return parser

def main(args):

    # --------------
    # INIT
    # --------------
    init_distributed_mode(args)
    fix_random_seeds(0)
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # --------------
    # GET BACKBONE
    # --------------
    print('Loading models...')
    # define clip model and preprocess pipeline, get input_dim and feature_dim
    clip_model, preprocess = clip.load(args.model)
    clip_model.float().eval()
    input_dim = clip_model.visual.input_resolution
    feature_dim = clip_model.visual.output_dim

    # --------------
    # GET COMBINER
    # --------------
    if args.combiner_mode == 'combiner_original':
        combiner = Combiner(clip_feature_dim=feature_dim, projection_dim=2560, hidden_dim=2 * 2560)
    elif args.combiner_mode in ('image_only', 'text_only', 'image_plus_text'):
        combiner = FeatureComb(args.combiner_mode, feature_comb_average=args.feature_comb_average)
    else:
        raise ValueError

    # --------------
    # LOAD PRETRAINED WEIGHTS
    # --------------
    if args.combiner_pretrain_path is not None:
        state_dict = torch.load(args.combiner_pretrain_path, map_location='cpu')
        state_dict = strip_state_dict(state_dict=state_dict, strip_key='module.')
        combiner.load_state_dict(state_dict)

    if args.clip_pretrain_path is not None:
        state_dict = torch.load(args.clip_pretrain_path, map_location='cpu')
        state_dict = strip_state_dict(state_dict=state_dict, strip_key='module.')
        clip_model.load_state_dict(state_dict)    

    # --------------
    # To cuda
    # --------------
    clip_model, combiner = clip_model.cuda(), combiner.cuda()
    if any([p.requires_grad for p in clip_model.parameters()]):
        clip_model = CLIPDistDataParallel(clip_model, device_ids=[args.gpu])
    if any([p.requires_grad for p in combiner.parameters()]):
        combiner = torch.nn.parallel.DistributedDataParallel(combiner, device_ids=[args.gpu])

    # --------------
    # GET DATASET
    # --------------
    print('Loading datasets...')
    tokenizer = partial(clip.tokenize, truncate=True)
    genecis_split_path = os.path.join(genecis_root, f'{args.dataset}.pkl')

    if args.dataset == 'CIRR':

        val_dataset_return_images = CIRRImageDataset(split='val', preprocess=preprocess, tokenizer=tokenizer)
        val_dataset_global = CIRRValGlobalDataset(split='val', preprocess=preprocess, tokenizer=tokenizer)

    elif args.dataset == 'mit_states':

        val_dataset_return_images = MITStatesImageOnly(split='test', transform=preprocess)
        val_dataset_global = MITStatesGlobalTestSet(split='test', transform=preprocess, tokenizer=tokenizer, use_complete_text_query=args.use_complete_text_query)

    elif 'attribute' in args.dataset:
        
        print(f'Evaluating on GeneCIS {args.dataset}')

        val_dataset_subset = VAWValSubset(val_split_path=genecis_split_path, tokenizer=tokenizer, transform=preprocess)

    elif 'object' in args.dataset:

        print(f'Evaluating on GeneCIS {args.dataset}')
        val_dataset_subset = COCOValSubset(val_split_path=genecis_split_path, tokenizer=tokenizer, transform=preprocess)

    else:

        raise ValueError

    # --------------
    # GET DATALOADER
    # --------------
    get_dataloader = partial(torch.utils.data.DataLoader, sampler=None,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False)

    if args.dataset in ('CIRR', 'mit_states'):
        valloader_global = get_dataloader(dataset=val_dataset_global)
        valloader_only_images = get_dataloader(dataset=val_dataset_return_images)
    else:
        valloader_subset = get_dataloader(dataset=val_dataset_subset)

    # --------------
    # EVALUTE
    # --------------
    # TODO: Adjust eval code for multiple GPUs
    if get_rank() == 0:
        
        if args.dataset in ('CIRR', 'mit_states'):
            validate_global(clip_model, combiner, valloader_only_images, valloader_global, topk=(1, 5, 10), save=args.pred_save_name)
        else:
            validate(clip_model, combiner, valloader_subset, topk=(1, 2, 3), save_path=args.pred_save_name)

if __name__ == '__main__':

    parser = argparse.ArgumentParser('CLIP4CIR', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
    