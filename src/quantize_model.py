import argparse
import random

import torch
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from datasets import load_dataset
from transformers import AutoTokenizer


def quantize_model(model_id, bits):
    out_dir = model_id + "-GPTQ"

    quantize_config = BaseQuantizeConfig(
        bits=bits,
        group_size=128,
        damp_percent=0.01,
        desc_act=False,
    )
    model = AutoGPTQForCausalLM.from_pretrained(model_id, quantize_config)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    n_samples = 1024
    data = load_dataset(
        "allenai/c4",
        data_files="en/c4-train.00001-of-01024.json.gz",
        split=f"train[:{n_samples*5}]",
    )
    tokenized_data = tokenizer("\n\n".join(data["text"]), return_tensors="pt")

    examples_ids = []
    for _ in range(n_samples):
        i = random.randint(
            0, tokenized_data.input_ids.shape[1] - tokenizer.model_max_length - 1
        )
        j = i + tokenizer.model_max_length
        input_ids = tokenized_data.input_ids[:, i:j]
        attention_mask = torch.ones_like(input_ids)
        examples_ids.append({"input_ids": input_ids, "attention_mask": attention_mask})

    model.quantize(
        examples_ids,
        batch_size=1,
        use_triton=True,
    )

    model.save_quantized(out_dir, use_safetensors=True)
    tokenizer.save_pretrained(out_dir)
    print("Model saved to", out_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Quantize a transformer model using GPTQ."
    )
    parser.add_argument(
        "--model_id",
        default="gpt2",
        type=str,
        required=True,
        help="The model identifier from Hugging Face model hub.",
    )
    parser.add_argument(
        "--bits",
        default=4,
        type=int,
        required=True,
        help="The number of bits to quantize the model to.",
    )
    args = parser.parse_args()
    quantize_model(args.model_id, args.bits)


if __name__ == "__main__":
    main()
