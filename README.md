# GPTQ Quantizer 

This small project demonstrates the quantization of transformer models using the `auto_gptq` library. It includes setup instructions, quantization of the model, and generation of text using the quantized model.

## Quickstart

Follow the instructions below to set up and run the project.

### Setup

1. **Clone the repository:**
   ```sh
   git clone https://github.com/Kirouane-Ayoub/GPTQ-Quantizer.git
   cd GPTQ-Quantizer
   ```

2. **Install the required packages:**
   ```sh
   pip install -r requirements.txt
   ```

### Usage

#### Quantize a Model

Run the quantization script with your desired model:
```sh
python src/quantize_model.py --model_id <model_id> --bits <bits>
```
Replace `<model_id>` with the identifier of the model you want to quantize (e.g., `gpt2`, `distilbert-base-uncased`, etc.).

Replace `<bits>` with the number of bits you want to use for quantization (e.g., `8`, `16`, etc.).

#### Load and Generate Text

To load the quantized model and generate text:
```sh
python src/load_and_generate.py --model_id <model_id> --text "Your input text" --max_length 100 
```
Replace `<model_id>` with the identifier of the model you quantized, and `"Your input text"` with the text you want to use as input for text generation.

Replace `100` with the desired maximum length of the generated text.
