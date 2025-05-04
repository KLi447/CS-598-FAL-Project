import logging
from typing import Tuple

from transformers import AutoModel

from mlora.model.llm import LlamaModel, LLMModel
from mlora.model.tokenizer import Tokenizer

MODEL_TYPE_DICT = {
    "llama": LlamaModel,
}


def load_partial_model(args) -> LLMModel:
    # load part of model to device
    assert args.rank != -1
    assert args.nodes >= args.rank

    logging.info(
        f"Pipeline parallelism, rank is {args.rank} and distributed over {args.nodes} nodes."
    )
    model = LlamaModel.from_pretrained(
        path=args.base_model,
        device=args.device,
        precision=args.precision,
    )
    seq_model = model.sequential()
    num_layers = len(seq_model)

    balance = [num_layers // args.nodes] * args.nodes
    for i in range(num_layers % args.nodes):
        balance[i] += 1

    partial_model_to_device = [
        index + sum(balance[: args.rank])
        for index in range(0, balance[args.rank])
    ]

    del model

    return MODEL_TYPE_DICT[args.model_type].from_pretrained(
        path=args.base_model,
        device=args.device,
        precision=args.precision,
        partial_model_to_device=partial_model_to_device,
    )


def load_full_model(args) -> LLMModel:
    return MODEL_TYPE_DICT[args.model_type].from_pretrained(
        path=args.base_model,
        device=args.device,
        precision=args.precision,
        partial_model_to_device=None,
    )


def load_model(args) -> Tuple[Tokenizer, LLMModel]:
    assert args.precision in ["nf4", "fp4", "int8", "bf16", "fp16", "fp32"]

    assert args.model_type in MODEL_TYPE_DICT, f"unkown model type {args.model_type}"

    tokenizer = Tokenizer(args.base_model)

    if args.pipeline:
        model = load_partial_model(args)
    else:
        model = load_full_model(args)

    return tokenizer, model
