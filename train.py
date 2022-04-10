import argparse
import itertools

from tensorboardX import SummaryWriter
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import yaml
from bisect import bisect

from numpy import random
import numpy as np
from visdialch.data.dataset import VisDialDataset

from torch.utils.data import DataLoader
from visdialch.encoders import Encoder
from visdialch.decoders import Decoder
from visdialch.metrics import SparseGTMetrics, NDCG, scores_to_ranks
from visdialch.model import EncoderDecoderModel
from visdialch.utils.checkpointing import CheckpointManager, load_checkpoint
from visdialch.data.vocabulary import Vocabulary

import json

# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config-yml", default="configs/default.yml",
    help="Path to a config file listing reader, model and solver parameters."
)
parser.add_argument(
    "--train-json", default="data/visdial_1.0_train.json",
    help="Path to json file containing VisDial v1.0 training data."
)
parser.add_argument(
    "--val-json", default="data/visdial_1.0_val.json",
    help="Path to json file containing VisDial v1.0 validation data."
)
parser.add_argument(
    "--val-dense-json", default="data/visdial_1.0_val_dense_annotations.json",
    help="Path to json file containing VisDial v1.0 validation dense ground truth annotations."
)
parser.add_argument(
    "--adj-train-h5", default="data/train_adj_list.h5",
    # "--adj-train-h5", default="data/debug_adj.h5",
    help="Path to pickle file containing adjacency matrices for each dialog."
)
parser.add_argument(
    "--adj-val-h5", default="data/adj_val_paths.h5",
    help="Path to pickle file containing adjacency matrices for each dialog."
)
parser.add_argument(
    "--adj-test-h5", default="data/adj_test_paths.h5",
    help="Path to pickle file containing adjacency matrices for each dialog."
)
parser.add_argument(
    "--captions-val-json", default="data/val2018.json",
    help="Path to json file containing VisDial v1.0 validation captions data."
)

parser.add_argument_group("Arguments independent of experiment reproducibility")
parser.add_argument(
    "--gpu-ids", nargs="+", type=int, default=0,
    help="List of ids of GPUs to use."
)
parser.add_argument(
    "--cpu-workers", type=int, default=4,
    help="Number of CPU workers for dataloader."
)
parser.add_argument(
    "--overfit", action="store_true",
    help="Overfit model on 5 examples, meant for debugging."
)
parser.add_argument(
    "--validate", action="store_true",
    help="Whether to validate on val split after every epoch."
)
parser.set_defaults(validate=False)
parser.add_argument(
    "--in-memory", action="store_true",
    help="Load the whole dataset and pre-extracted image features in memory. Use only in "
         "presence of large RAM, atleast few tens of GBs."
)
parser.add_argument(
    "--numberbatch", action="store_true",
    help="Use numberbatch instead of GloVe."
)

parser.add_argument(
    "--load-dialog", action="store_true",
    help="Load preprocessed the dialog. Else preprocess it in __init__ and __getitem__ of dataset."
)

parser.add_argument_group("Checkpointing related arguments")
parser.add_argument(
    "--save-dirpath", default="checkpoints/",
    help="Path of directory to create checkpoint directory and save checkpoints."
)
parser.add_argument(
    "--load-pthpath", default="",
    help="To continue training, path to .pth file of saved checkpoint."
)


# for reproducibility - refer https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# ================================================================================================
#   INPUT ARGUMENTS AND CONFIG
# ================================================================================================

args = parser.parse_args()

# keys: {"dataset", "model", "solver"}
config = yaml.load(open(args.config_yml))

if isinstance(args.gpu_ids, int): 
    args.gpu_ids = [args.gpu_ids]
device = torch.device("cuda", args.gpu_ids[0]) if args.gpu_ids[0] >= 0 else torch.device("cpu")
# device = torch.device("cpu")
torch.cuda.set_device(device)
# CUDA_LAUNCH_BLOCKING=1

# Print config and args.
print(yaml.dump(config, default_flow_style=False))
for arg in vars(args):
    print("{:<20}: {}".format(arg, getattr(args, arg)))

# ================================================================================================
#   SETUP DATASET, DATALOADER, MODEL, CRITERION, OPTIMIZER, SCHEDULER
# ================================================================================================

# ================================================================================================
# import nltk
# nltk.download('punkt')
# ================================================================================================

train_dataset = VisDialDataset(
    config["dataset"], 
    args.train_json,
    args.adj_train_h5,
    overfit=args.overfit, 
    in_memory=args.in_memory,
    num_workers=args.cpu_workers,
    return_options=True if config["model"]["decoder"] == "disc" else False,
    add_boundary_toks=False if config["model"]["decoder"] == "disc" else True,
    load_dialog = args.load_dialog
)
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=config["solver"]["batch_size"], 
    num_workers=args.cpu_workers, 
    shuffle=True
)

val_dataset = VisDialDataset(
    config["dataset"], 
    args.val_json,
    args.adj_val_h5,
    dense_annotations_jsonpath=args.val_dense_json, 
    overfit=args.overfit,
    in_memory=args.in_memory,
    num_workers=args.cpu_workers,
    return_options=True,
    add_boundary_toks=False if config["model"]["decoder"] == "disc" else True,
    load_dialog = args.load_dialog
)
val_dataloader = DataLoader(
    val_dataset, 
    batch_size=config["solver"]["batch_size"], 
    num_workers=args.cpu_workers
)


dataset_vocabulary = Vocabulary(
    config["dataset"]["word_counts_json"], min_count=config["dataset"]["vocab_min_count"]
)
# Read GloVe word embedding data
glove_token = torch.Tensor(np.load(config["dataset"]["glove_visdial_path"])).view(len(dataset_vocabulary), -1)

# Read ELMo word embedding data
elmo_token = torch.Tensor(np.load(config["dataset"]["elmo_visdial_path"])).view(len(dataset_vocabulary), -1)

ext_graph_vocabulary = Vocabulary(
    config["dataset"]["ext_word_counts_json"], min_count=0
)
numb_token = torch.Tensor(np.load(config["dataset"]["numberbatch_visdial_path"])).view(len(ext_graph_vocabulary), -1)

# Pass vocabulary to construct Embedding layer.
# if not args.numberbatch:
print('\n\nlen(ext_graph_vocabulary) = ', len(ext_graph_vocabulary))
encoder = Encoder(config["model"], train_dataset.vocabulary, ext_graph_vocabulary, glove_token, elmo_token, numb_token)
decoder = Decoder(config["model"], train_dataset.vocabulary, glove_token, elmo_token, False)
# decoder = Decoder(config["model"], train_dataset.vocabulary, ext_graph_vocabulary, glove_token, elmo_token, numb_token)
# decoder.w_embed = encoder.w_embed
# else:
#     encoder = Encoder(config["model"], train_dataset.vocabulary, numb_token, elmo_token, True)
#     decoder = Decoder(config["model"], train_dataset.vocabulary, numb_token, elmo_token, True)
#     decoder.w_embed = encoder.w_embed

print("Encoder: {}".format(config["model"]["encoder"]))
print("Decoder: {}".format(config["model"]["decoder"]))

# Share word embedding between encoder and decoder.
decoder.elmo_embed = encoder.elmo_embed
decoder.glove_embed = encoder.glove_embed
decoder.embed_change = encoder.embed_change

# Wrap encoder and decoder in a model
model = EncoderDecoderModel(encoder, decoder).to(device)
if -1 not in args.gpu_ids:
    model = nn.DataParallel(model, args.gpu_ids)


# Loss function.
if config["model"]["decoder"] == "disc":
    criterion = nn.CrossEntropyLoss()
elif config["model"]["decoder"] == "gen":
    criterion = nn.CrossEntropyLoss(
        ignore_index=train_dataset.vocabulary.PAD_INDEX
    )
else:
    raise NotImplementedError


if config["solver"]["training_splits"] == "trainval":
    iterations = (len(train_dataset) + len(val_dataset)) // config["solver"]["batch_size"] + 1
else:
    iterations = len(train_dataset) // config["solver"]["batch_size"] + 1


# lr_scheduler 1
def lr_lambda_fun(current_iteration: int) -> float:
    """
    Returns a learning rate multiplier.
    Till `warmup_epochs`, learning rate linearly increases to `initial_lr`,
    and then gets multiplied by `lr_gamma` every time a milestone is crossed.
    """

    current_epoch = float(current_iteration) / iterations
    if current_epoch <= config["solver"]["warmup_epochs"]:
        alpha = current_epoch / float(config["solver"]["warmup_epochs"])
        return config["solver"]["warmup_factor"] * (1. - alpha) + alpha
    else:
        idx = bisect(config["solver"]["lr_milestones"], current_epoch)
        return pow(config["solver"]["lr_gamma"], idx)


optimizer = optim.Adamax(model.parameters(), lr=float(config["solver"]["initial_lr"]))
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda_fun)
T = iterations * (config["solver"]["num_epochs"] - config["solver"]["warmup_epochs"] + 1)
# lr_scheduler 2
scheduler2 = lr_scheduler.CosineAnnealingLR(optimizer, int(T), eta_min=config["solver"]["eta_min"], last_epoch=-1)

# ================================================================================================
#   SETUP BEFORE TRAINING LOOP
# ================================================================================================

summary_writer = SummaryWriter(log_dir=args.save_dirpath)
checkpoint_manager = CheckpointManager(model, optimizer, args.save_dirpath, config=config)
sparse_metrics = SparseGTMetrics()
ndcg = NDCG()

# If loading from checkpoint, adjust start epoch and load parameters.
if args.load_pthpath == "":
    start_epoch = 0
else:
    # "path/to/checkpoint_xx.pth" -> xx
    start_epoch = int(args.load_pthpath.split("_")[-1][:-4]) + 1

    model_state_dict, optimizer_state_dict = load_checkpoint(args.load_pthpath)
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(model_state_dict)
    else:
        model.load_state_dict(model_state_dict)
    optimizer.load_state_dict(optimizer_state_dict)
    print("Loaded model from {}".format(args.load_pthpath))

# ================================================================================================
#   TRAINING LOOP
# ================================================================================================

# Forever increasing counter keeping track of iterations completed (for tensorboard logging).
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print('Number of parameters in model: ', count_parameters(model))
global_iteration_step = start_epoch * iterations

for epoch in range(start_epoch, config["solver"]["num_epochs"]):

    # --------------------------------------------------------------------------------------------
    #   ON EPOCH START  (combine dataloaders if training on train + val)
    # --------------------------------------------------------------------------------------------
    if config["solver"]["training_splits"] == "trainval":
        combined_dataloader = itertools.chain(train_dataloader, val_dataloader)
    else:
        combined_dataloader = itertools.chain(train_dataloader)

    print(f"\nTraining for epoch {epoch}:")
    for i, batch in enumerate(tqdm(combined_dataloader)):
        for key in batch:
            batch[key] = batch[key].to(device)
        optimizer.zero_grad()
        output = model(batch)
        target = (
            batch["ans_ind"]
            if config["model"]["decoder"] == "disc"
            else batch["ans_out"]
        )
        batch_loss = criterion(output.view(-1, output.size(-1)), target.view(-1))
        batch_loss.backward()
        optimizer.step()

        summary_writer.add_scalar("train/loss", batch_loss, global_iteration_step)
        summary_writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_iteration_step)
        
        
        if global_iteration_step <= iterations * config["solver"]["warmup_epochs"]:
            scheduler.step(global_iteration_step)
        else:
            global_iteration_step_in_2 = iterations * config["solver"]["warmup_epochs"] + 1 - global_iteration_step
            scheduler2.step(int(global_iteration_step_in_2))
        global_iteration_step += 1
    torch.cuda.empty_cache()

    # --------------------------------------------------------------------------------------------
    #   ON EPOCH END  (checkpointing and validation)
    # --------------------------------------------------------------------------------------------
    checkpoint_manager.step()

    # Validate and report automatic metrics.
    if args.validate:

        # Switch dropout, batchnorm etc to the correct mode.
        model.eval()

        print(f"\nValidation after epoch {epoch}:")
        for i, batch in enumerate(tqdm(val_dataloader)):
            for key in batch:
                batch[key] = batch[key].to(device)
            with torch.no_grad():
                output = model(batch)
            sparse_metrics.observe(output, batch["ans_ind"])
            if "gt_relevance" in batch:
                output = output[torch.arange(output.size(0)), batch["round_id"] - 1, :]
                ndcg.observe(output, batch["gt_relevance"])

        all_metrics = {}
        all_metrics.update(sparse_metrics.retrieve(reset=True))
        all_metrics.update(ndcg.retrieve(reset=True))
        for metric_name, metric_value in all_metrics.items():
            print(f"{metric_name}: {metric_value}")
        summary_writer.add_scalars("metrics", all_metrics, global_iteration_step)

        model.train()
        torch.cuda.empty_cache()
