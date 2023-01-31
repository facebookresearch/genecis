import os
import sys

# Only add this import in case we are running from a standard bash script
# Submitit requires sys.path to be done inside the job
if 'SLURM_PROCID' not in os.environ:
    sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

# Combiner model
from models.combiner_model import Combiner

# Datasets
from datasets.cirr_datasets import CIRRImageDataset, CIRRValGlobalDataset
from datasets.cc_3m_dataset import CCConditionalDistractor
from datasets.vaw_dataset import VAWValSubset
from datasets.coco_dataset import COCOValSubset

# Train and eval functions
from train.epoch_train_functions import train_one_epoch, val_one_epoch
from eval.eval_functions import validate, validate_global

# CLIP
import clip
import torch

# Utils
import argparse
import torch.backends.cudnn as cudnn
from utils.dist_utils import fix_random_seeds, init_distributed_mode, get_rank, CLIPDistDataParallel, worker_init_fn
from utils.gen_utils import bool_flag, none_flag, step_scheduler, convert_models_to_fp32, restart_from_checkpoint, get_mean_lr, create_uq_log_dir, _to_rgb, set_requires_grad_clip
from functools import partial

from torchvision import transforms as T
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import config as cfg

def get_args_parser():

    parser = argparse.ArgumentParser('CC3M', add_help=False)

    parser.add_argument('--backbone', default='RN50x4', type=str, help='Which CLIP model we are using as backbone')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--log_dir', default=cfg.log_dir)
    parser.add_argument('--seed', default=0, type=int)

    # Optimization hyper params
    parser.add_argument('--finetune_mode', default='image_plus_text_whole', type=none_flag, help='\{image_plus_text, text_only, image_only, text_only_whole\}')
    parser.add_argument('--optimizer', default='adamw', type=str)
    parser.add_argument('--scheduler', default='cosine', type=str)
    parser.add_argument('--batch_size_per_gpu', default=256, type=int)
    parser.add_argument('--lr', default=1e-6, type=float)
    parser.add_argument('--lamda', default=100, type=float, help='1 / temperature for sotftmax operation')
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--clip_grad_norm', default=True, type=bool_flag)

    # Loss lambdas
    parser.add_argument('--base_contrastive', default=1, type=float)

    # CC3M extracted data paths
    parser.add_argument('--cc3m_deterministic_root_path', default=cfg.cc3m_deterministic_root_path, type=int)
    parser.add_argument('--cc3m_annots_path', default=cfg.cc3m_tsg_path, type=str)

    # General CC3M params
    parser.add_argument('--min_images_with_subject', default=cfg.cc3m_min_images_with_subject, type=int)
    parser.add_argument('--num_samples_per_epoch', default=1600000, type=int)
    parser.add_argument('--deterministic_samples_key', default=None, type=none_flag, help='A key defining which deterministic samples to use')
    parser.add_argument('--concreteness_threshold', default=cfg.cc3m_concreteness_threshold, type=float, help="Threshold for how visually conrete the sampls images are")
    parser.add_argument('--val_start_idx', default=0, type=int, help='Dictates where from the dataset to sample the validation set')

    # Dist params
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")

    # Combiner Model
    parser.add_argument('--combiner_mode', default='combiner_original', type=str, help='\{text_only, image_only, image_plus_text, combiner_original}')

    return parser

def main(args):
    
    # --------------
    # INIT
    # --------------
    args.writer = SummaryWriter(log_dir=args.log_dir)
    args.chkpt_path = os.path.join(args.log_dir, 'checkpoint.pt')

    init_distributed_mode(args)
    fix_random_seeds(args.seed)
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # --------------
    # GET BACKBONE
    # --------------
    print('Loading models...')
    
    # define clip model and preprocess pipeline, get input_dim and feature_dim
    clip_model, clip_preprocess = clip.load(args.backbone)
    image_size = clip_preprocess.transforms[1].size[0]

    train_prepocess = T.Compose([
        T.Resize(size=int(image_size / 0.875)),
        T.RandomResizedCrop(size=image_size, scale=(0.9, 1)),
        _to_rgb,
        T.ToTensor(),
        T.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711))
    ])

    convert_models_to_fp32(clip_model)
    clip_model.eval()

    input_dim = clip_model.visual.input_resolution
    feature_dim = clip_model.visual.output_dim

    # Turn grad off in all parameters in backbone
    for p in clip_model.parameters():
        p.requires_grad = False

    # --------------
    # GET COMBINER
    # --------------
    combiner = Combiner(clip_feature_dim=feature_dim, projection_dim=2560, hidden_dim=2 * 2560, norm_feats_before_combining=args.norm_feats_before_combining)

    # --------------
    # FINE TUNEBACKBONE
    # --------------
    set_requires_grad_clip(args.finetune_mode, clip_model)

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
    # Dataset class changes based on which loss we are using
    print('Loading datasets...')
    tokenizer = partial(clip.tokenize, truncate=True)
    train_dataset = CCConditionalDistractor(cc3m_annots_path=args.cc3m_annots_path,
        min_images_with_subject=args.min_images_with_subject, 
        num_samples_per_epoch=args.num_samples_per_epoch,
        cc3m_deterministic_root_path=args.cc3m_deterministic_root_path,
        transform=train_prepocess, 
        tokenizer=tokenizer)

    # Get val dataset
    val_dataset_cc3m = train_dataset.split_to_train_val(val_preprocess=clip_preprocess, val_start_idx=args.val_start_idx)

    # --------------
    # GET DOWNSTREAM EVAL DATASETS
    # CIRR and Change Attribute and Focus Object
    # --------------
    cirr_val_dataset_return_images = CIRRImageDataset(split='val', preprocess=clip_preprocess, tokenizer=tokenizer)
    cirr_val_dataset_global = CIRRValGlobalDataset(split='val', preprocess=clip_preprocess, tokenizer=tokenizer)
    
    change_attribute_split_path = os.path.join(cfg.genecis_root, 'change_attribute.pkl')
    print(f'Evaluating on GeneCIS {change_attribute_split_path}')
    change_attribute_valset = VAWValSubset(val_split_path=change_attribute_split_path, tokenizer=tokenizer, transform=clip_preprocess)

    same_object_split_path = os.path.join(cfg.genecis_root, 'same_object.pkl')
    print(f'Evaluating on GeneCIS {args.dataset}')
    same_object_valset = COCOValSubset(val_split_path=same_object_split_path, tokenizer=tokenizer, transform=clip_preprocess)

    # --------------
    # DATALOADERS
    # --------------
    sampler = torch.utils.data.DistributedSampler(train_dataset, shuffle=True)
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=True,
        drop_last=False,
    )

    cirr_valloader_only_images = torch.utils.data.DataLoader(
        cirr_val_dataset_return_images,
        sampler=None,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    cirr_valloader_global = torch.utils.data.DataLoader(
        cirr_val_dataset_global,
        sampler=None,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    change_attribute_valloader = torch.utils.data.DataLoader(
        change_attribute_valset,
        sampler=None,
        batch_size=int(args.batch_size_per_gpu / 8),
        num_workers=args.num_workers,
        pin_memory=True,
    )

    same_object_valloader = torch.utils.data.DataLoader(
        same_object_valset,
        sampler=None,
        batch_size=int(args.batch_size_per_gpu / 8),
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # Val dataset for CC3M
    valloader_cc3m = torch.utils.data.DataLoader(
        val_dataset_cc3m,
        sampler=torch.utils.data.DistributedSampler(val_dataset_cc3m, shuffle=False),
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=True,
        drop_last=True,
    )

    # --------------
    # OPTIMIZER
    # --------------
    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(lr=args.lr, weight_decay=args.weight_decay,
        params=list(combiner.parameters()) + list(clip_model.parameters()))
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(lr=args.lr, weight_decay=args.weight_decay,
        params=list(combiner.parameters()) + list(clip_model.parameters()))
    else:
        raise ValueError

    # --------------
    # Scaler
    # --------------
    scaler = torch.cuda.amp.GradScaler()
    args.scaler = scaler

    # --------------
    # SCHEDULER
    # --------------
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 1e-3)

    # --------------
    # LOAD FROM CHECKPOINT
    # --------------
    to_restore = {
        'epoch': 0,
        'best_val_recall': 0
    }
    restart_from_checkpoint(ckp_path=args.chkpt_path,
        run_variables=to_restore,
        clip_model=clip_model, 
        combiner=combiner,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=args.scaler
    )
    best_val_recall = to_restore['best_val_recall']
    start_epoch = to_restore['epoch']

    # --------------
    # TRAIN
    # --------------

    print(f'Starting training from epoch {start_epoch}...')
    for epoch in range(start_epoch, args.epochs):

        sampler.set_epoch(epoch)

        train_loss_meters = train_one_epoch(clip_model=clip_model, combiner=combiner,
        trainloader=trainloader, optimizer=optimizer, args=args)

        # --------------
        # Evaluate and log
        # --------------
        global_step = len(trainloader) * (epoch + 1)

        val_loss_meters = val_one_epoch(clip_model=clip_model, combiner=combiner,
        valloader=valloader_cc3m, args=args, recall_topk=(1, 5, 10))

        if get_rank() == 0:

            # Val epoch CIRR
            recall_meters_cirr = validate_global(clip_model, combiner, cirr_valloader_only_images, cirr_valloader_global, topk=(1, 5, 10))

            # Val epoch Change Attribute
            recall_meters_att = validate(clip_model, combiner, change_attribute_valloader, topk=(1, 2, 3))

            # Val epoch Focus Object
            recall_meters_obj = validate(clip_model, combiner, same_object_valloader, topk=(1, 2, 3))

            # Log losses
            for loss_name in train_loss_meters.keys():
                if 'combiner' not in loss_name:
                    args.writer.add_scalars(f'CC3M Metrics/{loss_name}', {
                        'Train': train_loss_meters[loss_name],
                        'Val': val_loss_meters[loss_name]
                    },
                    global_step=global_step)

            # Log Val recalls
            val_recall_logs = {f'Recall @ {k}': val_loss_meters[f'Recall @ {k}'] for k in (1, 5, 10)}
            args.writer.add_scalars(f'CC3M Metrics/Val Recall', val_recall_logs, global_step=global_step)

            # Log CIRR recalls
            for recall_at, meter in recall_meters_cirr.items():
                args.writer.add_scalar(f'CIRR Recall/Recall @ {recall_at}', meter.avg,global_step=global_step)

            # Log VAW recalls
            for recall_at, meter in recall_meters_att.items():
                args.writer.add_scalar(f'Change Attribute Recall/Recall @ {recall_at}', meter.avg,global_step=global_step)

            # Log COCO recalls
            for recall_at, meter in recall_meters_obj.items():
                args.writer.add_scalar(f'Focus Object Recall/Recall @ {recall_at}', meter.avg,global_step=global_step)
            
            # Log LR
            args.writer.add_scalar('LR', scalar_value=get_mean_lr(optimizer), global_step=global_step)

            # Log misc 
            args.writer.add_scalars(f'z-Combiner Feat Dist', {
                    'Combined-text': train_loss_meters['combiner_text_feat_dist'],
                    'Combined-image': train_loss_meters['combiner_image_feat_dist']
                },
                global_step=global_step)

            # Print
            print(f'Epoch {epoch} | Train Loss: {train_loss_meters["total_loss"]:.4f}')
            val_print_str = f'Epoch {epoch} CIRR Eval |' + ' | '.join([f'Recall @ {k} = {v.avg:.4f}' for k, v in recall_meters_cirr.items()])
            print(val_print_str)

            # Save
            save_dict = {
                'clip_model': clip_model.state_dict(),
                'combiner': combiner.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch + 1,
                'scaler': args.scaler.state_dict(),
                'best_val_recall': best_val_recall
            }

            print(f'Saving checkpoint to {args.chkpt_path}...')
            torch.save(save_dict, args.chkpt_path)

            if recall_meters_cirr[1].avg > best_val_recall:

                print(f'Saving best models to {args.log_dir}...')
                torch.save(clip_model.state_dict(), os.path.join(args.log_dir, 'clip_model_best.pt'))
                torch.save(combiner.state_dict(), os.path.join(args.log_dir, 'combiner_best.pt'))

                best_val_recall = recall_meters_cirr[1].avg

        np.random.seed(np.random.get_state()[1][0] + epoch)
        step_scheduler(scheduler=scheduler, metric=train_loss_meters['total_loss'], args=args)

if __name__ == '__main__':

    parser = argparse.ArgumentParser('CC3M', parents=[get_args_parser()])
    args = parser.parse_args()

    # Create random log file
    args.log_dir = create_uq_log_dir(args.log_dir)

    main(args)
    