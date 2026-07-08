# split_dataset.py
# Usage: python split_dataset.py out/ [num_eval_sites]
#
# Randomly holds out a few sites entirely for evaluation -- these are never
# touched during training, so later you can honestly check whether
# fine-tuning helped on data the model has genuinely never seen.

import sys
import os
import json
import random

out_dir = sys.argv[1] if len(sys.argv) > 1 else "out"
num_eval = int(sys.argv[2]) if len(sys.argv) > 2 else 4

sites = sorted([
    d for d in os.listdir(out_dir)
    if os.path.isdir(os.path.join(out_dir, d))
    and os.path.exists(os.path.join(out_dir, d, "tree.json"))
])

if len(sites) <= num_eval:
    print(f"Only {len(sites)} sites found, need more than {num_eval} to hold any out.")
    sys.exit(1)

random.seed(42)  # fixed seed so this split is reproducible if you re-run it
random.shuffle(sites)

eval_sites = sites[:num_eval]
train_sites = sites[num_eval:]

with open("train_sites.json", "w") as f:
    json.dump(train_sites, f, indent=2)
with open("eval_sites.json", "w") as f:
    json.dump(eval_sites, f, indent=2)

print(f"Total sites: {len(sites)}")
print(f"Train: {len(train_sites)} -> train_sites.json")
print(f"Eval (held out): {len(eval_sites)} -> eval_sites.json")
print("\nHeld-out eval sites:")
for s in eval_sites:
    print(f"  {s}")
