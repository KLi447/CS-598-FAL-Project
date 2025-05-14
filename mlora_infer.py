#!/usr/bin/env python3
"""
mlora_infer.py

This script loads the full base model (non-pipeline) along with a merged adapter checkpoint,
applies the adapter weights, and then runs inference on a provided prompt.

Usage:
    python mlora_infer.py --base_model TinyLlama/TinyLlama-1.1B-Chat-v0.4 \
        --model_type llama --precision fp32 --merged_adapter_ckpt adapters/lora_sft_0/adapter_model.bin \
        --prompt "Could you provide an introduction to mLoRA?" --device cuda
"""

import argparse
import os
import torch
import uuid

import mlora.model
from mlora.model.tokenizer import Tokenizer
from mlora.model.loader import load_full_model  # Loads the complete base model.
from mlora.model.args import ModelData

def parse_args():
    parser = argparse.ArgumentParser(description="Inference using merged mLoRA adapter.")
    parser.add_argument("--base_model", type=str, required=True,
                        help="Base model path or name (e.g., TinyLlama/TinyLlama-1.1B-Chat-v0.4)")
    parser.add_argument("--model_type", type=str, default="llama",
                        help="Model type (e.g., llama)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run inference on (cuda or cpu)")
    parser.add_argument("--precision", type=str, default="fp32",
                        choices=["nf4", "fp4", "int8", "bf16", "fp16", "fp32"],
                        help="Model precision")
    parser.add_argument("--merged_adapter_ckpt", type=str, required=True,
                        help="Path to the merged adapter checkpoint (e.g., adapters/lora_sft_0/adapter_model.bin)")
    parser.add_argument("--prompt", type=str, default="Hello, how are you?",
                        help="Input prompt for inference")
    return parser.parse_args()

def apply_merged_adapter_to_full_model(full_model, merged_adapter_dict):
    """
    Merge the adapter parameters into the full model.
    The merging logic below assumes that keys in merged_adapter_dict correspond to
    parameters in the full model. Adjust if you need to handle layer offsets.
    """
    if hasattr(full_model, "state_dict"):
        state = full_model.state_dict()
        for key, value in merged_adapter_dict.items():
            if key in state:
                state[key].copy_(value)
            else:
                print(f"Warning: adapter key {key} not found in full model state_dict.")
        full_model.load_state_dict(state)
    else:
        for key, value in merged_adapter_dict.items():
            if key in full_model.__dict__:
                attr = full_model.__dict__[key]
                if isinstance(attr, torch.Tensor):
                    attr.copy_(value)
                else:
                    full_model.__dict__[key] = value
            else:
                print(f"Warning: adapter key {key} not found in model.__dict__.")
    return full_model

def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() or args.device=="cpu" else "cpu"
    
    # Instantiate the tokenizer.
    tokenizer = Tokenizer(args.base_model)
    
    # Create a dummy args namespace for loading the full model.
    import argparse
    dummy_args = argparse.Namespace(
        base_model=args.base_model,
        model_type=args.model_type,
        device=device,
        precision=args.precision,
        pipeline=False  # We want the full model for inference.
    )
    
    # Load the full base model.
    full_model = load_full_model(dummy_args)
    
    # Transfer model to device.
    if hasattr(full_model, "to"):
        full_model.to(args.device)
    elif args.device=="cuda" and hasattr(full_model, "cuda"):
        full_model = full_model.cuda()
    
    # Set model to evaluation mode if available.
    if hasattr(full_model, "eval"):
        full_model.eval()
    else:
        print("Warning: Model does not implement eval().")
    
    # Load the merged adapter checkpoint.
    merged_adapter = torch.load(args.merged_adapter_ckpt, map_location="cpu")
    
    # Apply adapter weights to the full model.
    full_model = apply_merged_adapter_to_full_model(full_model, merged_adapter)
    
    # Prepare the input prompt:
    # Tokenize the prompt and create tensors.
    token_ids = tokenizer.encode(args.prompt)
    batch_tokens = torch.tensor([token_ids], dtype=torch.int64, device=args.device)
    batch_mask = torch.zeros_like(batch_tokens, dtype=torch.float32, device=args.device)
    
    # Create a ModelData instance as required by the forward() call.
    data_obj = ModelData(
        batch_tokens_=batch_tokens,
        batch_mask_=batch_mask,
        enable_checkpoint_=False,
        data_config_=[],
        random_id_=uuid.uuid4().int,
        task_name_="inference"
    )
    
    # Run inference.
    with torch.no_grad():
        outputs = full_model.forward(data_obj)
    
    # Assume outputs are logits; get the predicted token IDs.
    predicted_ids = outputs.argmax(dim=-1).squeeze().tolist()
    output_text = tokenizer.decode(predicted_ids)
    
    print("Input Prompt:", args.prompt)
    print("Generated Output:", output_text)

if __name__ == "__main__":
    main()
