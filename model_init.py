# requirements: transformers, torch

from pathlib import Path 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Model names and save paths for different sizes of BART models
small_model_name = "gotutiyan/gec-bart-base" # smaller, suitable for limited resources
small_save_path = "./gec-bart-base"

large_model_name = "gotutiyan/gec-bart-large" # better performance, requires more resources 
large_save_path = "./gec-bart-large"

def initialise_model(model_name, save_path):
    
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Load the tokenizer and model from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Save locally
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)

    print("Model downloaded and saved to:", save_path)

    print("BART downloaded successfully")
    return tokenizer, model

# initialise small BART model 
def initialise_small_model(): 
    return initialise_model(small_model_name, small_save_path)

# initialise large BART model 
def initialise_large_model(): 
    return initialise_model(large_model_name, large_save_path)