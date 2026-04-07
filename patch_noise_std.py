"""Patch a RSL-RL checkpoint to reset noise_std to a higher value.

Usage:
    python patch_noise_std.py <checkpoint_path> [--std 1.0]

Creates a new file alongside the original: <name>_patched.pt
"""

import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint", type=str)
parser.add_argument("--std", type=float, default=1.0)
args = parser.parse_args()

ckpt = torch.load(args.checkpoint, map_location="cpu")

print("Checkpoint keys:", list(ckpt.keys()))

if "model_state_dict" in ckpt:
    state = ckpt["model_state_dict"]
    std_keys = [k for k in state if "std" in k.lower()]
    print(f"std-related keys: {std_keys}")
    for k in std_keys:
        old_val = state[k]
        state[k] = torch.full_like(old_val, args.std)
        print(f"  {k}: {old_val.mean().item():.4f} -> {args.std}")

out_path = args.checkpoint.replace(".pt", "_patched.pt")
torch.save(ckpt, out_path)
print(f"\nSaved patched checkpoint to: {out_path}")
