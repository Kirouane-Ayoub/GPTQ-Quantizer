import argparse

import torch
from auto_gptq import AutoGPTQForCausalLM
from transformers import AutoTokenizer, pipeline


def load_and_generate(model_id, prompt, max_length):
    out_dir = model_id + "-GPTQ"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = AutoGPTQForCausalLM.from_quantized(
        out_dir,
        device=device,
        use_triton=True,
        use_safetensors=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(out_dir)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    result = generator(prompt, do_sample=True, max_length=max_length)[0][
        "generated_text"
    ]
    print("Generated text:", result)


def main():
    parser = argparse.ArgumentParser(
        description="Load a quantized model and generate text."
    )
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="The model identifier from Hugging Face model hub.",
    )
    parser.add_argument(
        "--prompt", type=str, required=True, help="The input text to generate from."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=200,
        help="The maximum length of the generated text.",
    )
    args = parser.parse_args()
    load_and_generate(args.model_id, args.prompt)


if __name__ == "__main__":
    main()
