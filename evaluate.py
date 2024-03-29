import argparse
import json
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

import numpy as np
from numpy import random
from visdialch.data.vocabulary import Vocabulary
import json

from visdialch.data.dataset import VisDialDataset
from visdialch.encoders import Encoder
from visdialch.decoders import Decoder
from visdialch.metrics import SparseGTMetrics, NDCG, scores_to_ranks
from visdialch.model import EncoderDecoderModel
from visdialch.utils.checkpointing import load_checkpoint


parser = argparse.ArgumentParser("Evaluate and/or generate EvalAI submission file.")
parser.add_argument(
    "--config-yml", default="configs/default.yml",
    help="Path to a config file listing reader, model and optimization parameters."
)
parser.add_argument(
    "--split", default="val", choices=["val", "test"],
    help="Which split to evaluate upon."
)
parser.add_argument(
    "--val-json", default="data/visdial_1.0_val.json",
    help="Path to VisDial v1.0 val data. This argument doesn't work when --split=test."
)
parser.add_argument(
    "--val-dense-json", default="data/visdial_1.0_val_dense_annotations.json",
    help="Path to VisDial v1.0 val dense annotations (if evaluating on val split)."
         "This argument doesn't work when --split=test."
)
parser.add_argument(
    "--test-json", default="data/visdial_1.0_test.json",
    help="Path to VisDial v1.0 test data. This argument doesn't work when --split=val."
)
parser.add_argument(
    "--adj-val-h5", default="data/val_adj_list.h5",
    help="Path to pickle file containing adjacency matrices for each dialog."
)
parser.add_argument(
    "--adj-test-h5", default="data/test_adj_list.h5",
    help="Path to pickle file containing adjacency matrices for each dialog."
)


parser.add_argument(
    "--captions-test-json", default="data/test2018.json",
    help="Path to json file containing VisDial v1.0 test captions data."
)
parser.add_argument(
    "--captions-val-json", default="data/val2018.json",
    help="Path to json file containing VisDial v1.0 validation captions data."
)


parser.add_argument_group("Evaluation related arguments")
parser.add_argument(
    "--load-pthpath", default="checkpoints/checkpoint_xx.pth",
    help="Path to .pth file of pretrained checkpoint."
)

parser.add_argument_group("Arguments independent of experiment reproducibility")
parser.add_argument(
    "--gpu-ids", nargs="+", type=int, default=-1,
    help="List of ids of GPUs to use."
)
parser.add_argument(
    "--cpu-workers", type=int, default=4,
    help="Number of CPU workers for reading data."
)
parser.add_argument(
    "--overfit", action="store_true",
    help="Overfit model on 5 examples, meant for debugging."
)
parser.add_argument(
    "--in-memory", action="store_true",
    help="Load the whole dataset and pre-extracted image features in memory. Use only in "
         "presence of large RAM, atleast few tens of GBs."
)

parser.add_argument_group("Submission related arguments")
parser.add_argument(
    "--save-ranks-path", default="logs/ranks.json",
    help="Path (json) to save ranks, in a EvalAI submission format."
)
parser.add_argument(
    "--numberbatch", action="store_true",
    help="Use numberbatch instead of GloVe."
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

if isinstance(args.gpu_ids, int): args.gpu_ids = [args.gpu_ids]
device = torch.device("cuda", args.gpu_ids[0]) if args.gpu_ids[0] >= 0 else torch.device("cpu")

# Print config and args.
print(yaml.dump(config, default_flow_style=False))
for arg in vars(args):
    print("{:<20}: {}".format(arg, getattr(args, arg)))


# ================================================================================================
#   SETUP DATASET, DATALOADER, MODEL
# ================================================================================================

if args.split == "val":
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
        val=True
        # load_dialog = args.load_dialog
    )
else:
    val_dataset = VisDialDataset(
        config["dataset"], 
        args.test_json,
        args.adj_test_h5,
        overfit=args.overfit, 
        in_memory=args.in_memory,
        num_workers=args.cpu_workers,
        return_options=True if config["model"]["decoder"] == "disc" else False,
        add_boundary_toks=False if config["model"]["decoder"] == "disc" else True
        # load_dialog = args.load_dialog
    )
val_dataloader = DataLoader(
    val_dataset, batch_size=config["solver"]["batch_size"], num_workers=args.cpu_workers
)

ext_graph_vocabulary = Vocabulary(
config["dataset"]["ext_word_counts_json"], min_count=0
)
dataset_vocabulary = Vocabulary(
    config["dataset"]["word_counts_json"], min_count=config["dataset"]["vocab_min_count"]
)

# Read GloVe word embedding data
glove_token = torch.Tensor(np.load(config["dataset"]["glove_visdial_path"])).view(len(dataset_vocabulary), -1)

# Read ELMo word embedding data
elmo_token = torch.Tensor(np.load(config["dataset"]["elmo_visdial_path"])).view(len(dataset_vocabulary), -1)

numb_token = torch.Tensor(np.load(config["dataset"]["numberbatch_visdial_path"])).view(len(ext_graph_vocabulary), -1)

# Pass vocabulary to construct Embedding layer.

encoder = Encoder(config["model"], val_dataset.vocabulary, ext_graph_vocabulary, glove_token, elmo_token, numb_token)
decoder = Decoder(config["model"], val_dataset.vocabulary, glove_token, elmo_token, False)
decoder.elmo_embed = encoder.elmo_embed
decoder.glove_embed = encoder.glove_embed
decoder.embed_change = encoder.embed_change
print("Encoder: {}".format(config["model"]["encoder"]))
print("Decoder: {}".format(config["model"]["decoder"]))


# Wrap encoder and decoder in a model.
model = EncoderDecoderModel(encoder, decoder).to(device)
if -1 not in args.gpu_ids:
    model = nn.DataParallel(model, args.gpu_ids)

model_state_dict, _ = load_checkpoint(args.load_pthpath)
if isinstance(model, nn.DataParallel):
    model.module.load_state_dict(model_state_dict)
else:
    model.load_state_dict(model_state_dict)
print("Loaded model from {}".format(args.load_pthpath))

# Declare metric accumulators (won't be used if --split=test)
sparse_metrics = SparseGTMetrics()
ndcg = NDCG()

# ================================================================================================
#   EVALUATION LOOP
# ================================================================================================

model.eval()
ranks_json = []

for _, batch in enumerate(tqdm(val_dataloader)):
    for key in batch:
        batch[key] = batch[key].to(device)
    with torch.no_grad():
        output = model(batch)

    ranks = scores_to_ranks(output)
    for i in range(len(batch["img_ids"])):
        # Cast into types explicitly to ensure no errors in schema.
        # Round ids are 1-10, not 0-9
        if args.split == "test":
            ranks_json.append({
                "image_id": batch["img_ids"][i].item(),
                "round_id": int(batch["num_rounds"][i].item()),
                "ranks": [rank.item() for rank in ranks[i][batch["num_rounds"][i] - 1]]
            })
        else:
            for j in range(batch["num_rounds"][i]):
                ranks_json.append({
                    "image_id": batch["img_ids"][i].item(),
                    "round_id": int(j + 1),
                    "ranks": [rank.item() for rank in ranks[i][j]]
                })

print("Writing ranks to {}".format(args.save_ranks_path))
os.makedirs(os.path.dirname(args.save_ranks_path), exist_ok=True)
json.dump(ranks_json, open(args.save_ranks_path, "w"))