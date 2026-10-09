# requirements: datasets 
from datasets import Dataset 

# given .src and .tgt filepaths, create the dataset 
def create_dataset(source_file, target_file): 
    
    # read source file 
    with open(source_file, "r", encoding="utf-8") as source_file:
        source_sentences = [line.strip() for line in source_file if line.strip()]

    # read target file 
    with open(target_file, "r", encoding="utf-8") as target_file:
        target_sentences = [line.strip() for line in target_file if line.strip()]

    # check that the files are aligned
    if len(source_sentences) != len(target_sentences):
        raise ValueError(f"Number of source sentences ({len(source_sentences)}) does not match number of target sentences ({len(target_sentences)}).")

    # create dataset
    dataset = Dataset.from_dict({
        "source": source_sentences,
        "target": target_sentences
    })

    print("dataset created.")
    return dataset 

# uncomment below to create training datasets 
""" 
# create FCE v2.1 dataset 
fce_dataset = create_dataset(fce_source, fce_target)

# create Lang-8 Corpus of Learner English dataset 
lang8_dataset = create_dataset(lang8_source, lang8_target)

# create NUCLE (NUS Corpus of Learner English) dataset 
nucle_dataset = create_dataset(nucle_source, nucle_target)

# create W&I+LOCNESS v2.1 dataset 
wi_dataset = create_dataset(wi_source, wi_target)
"""

# tokenize the batch of data passed into the function 
def tokenize_data_batch(tokenizer, data_batch):
    # tokenize inputs and labels 
    inputs = tokenizer(data_batch["source"], max_length=128, truncation=True)
    labels = tokenizer(data_batch["target"], max_length=128, truncation=True)

    # add correct target output to tokenized input data 
    inputs["labels"] = labels["input_ids"]
    
    return inputs 

# converts text to tokenized examples from the dataset  
def tokenize_dataset(dataset): 
    # applies tokenize_data_batch to the dataset passed into the function  
    tokenized_dataset = dataset.map(
        tokenize_data_batch,
        batched=True,
        remove_columns=["source", "target"]
    )

    return tokenized_dataset
