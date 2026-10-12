# requirements: datasets 
from datasets import Dataset 

# filepaths to .src and .tgt files for FCE v2.1 dataset 
fce_source = "fce-src-tgt/fce.train.gold.bea19.src"
fce_target = "fce-src-tgt/fce.train.gold.bea19.tgt"

# filepaths to .src and .tgt files for Lang-8 Corpus of Learner English dataset 
lang8_source = r"" 
lang8_target = r""

# filepaths to .src and .tgt files for NUCLE (NUS Corpus of Learner English) dataset 
nucle_source = "nucle-src-tgt/nucle.train.gold.bea19.src"
nucle_target = "nucle-src-tgt/nucle.train.gold.bea19.tgt"

# filepaths to .src and .tgt files for W&I+LOCNESS v2.1 dataset
wi_source = r""
wi_target = r""

# given .src and .tgt filepaths, create the dataset 
def create_dataset(source_file, target_file): 

    # read source file 
    with open(source_file, "r", encoding="utf-8") as source_file:
        source_sentences = [line.rstrip("\r\n") for line in source_file]

    # read target file 
    with open(target_file, "r", encoding="utf-8") as target_file:
        target_sentences = [line.rstrip("\r\n") for line in target_file]

    # check that the files are aligned
    if len(source_sentences) != len(target_sentences):
        raise ValueError(f"Number of source sentences ({len(source_sentences)}) does not match number of target sentences ({len(target_sentences)}).")

    # create dataset
    dataset = Dataset.from_dict({
        "source": source_sentences,
        "target": target_sentences
    })

    return dataset 

# tokenize the batch of data passed into the function 
def tokenize_data_batch(tokenizer, data_batch):
    # tokenize inputs and labels 
    inputs = tokenizer(data_batch["source"], max_length=128, truncation=True)
    labels = tokenizer(data_batch["target"], max_length=128, truncation=True)

    # add correct target output to tokenized input data 
    inputs["labels"] = labels["input_ids"]
    
    return inputs 

# converts text to tokenized examples from the dataset  
def tokenize_dataset(tokenizer, dataset): 
    # applies tokenize_data_batch to the dataset passed into the function  
    tokenized_dataset = dataset.map(
        lambda data_batch: tokenize_data_batch(tokenizer, data_batch),
        batched=True,
        remove_columns=["source", "target"]
    )

    return tokenized_dataset

# create all datasets from their respective .src and .tgt files 
def create_all_datasets(): 
    # create FCE v2.1 dataset 
    fce_dataset = create_dataset(fce_source, fce_target)
    print("fce_dataset created.")

    # create Lang-8 Corpus of Learner English dataset 
    lang8_dataset = create_dataset(lang8_source, lang8_target)
    print("lang8_dataset created.")
    
    # create NUCLE (NUS Corpus of Learner English) dataset 
    nucle_dataset = create_dataset(nucle_source, nucle_target)
    print("nucle_dataset created.")
    
    # create W&I+LOCNESS v2.1 dataset 
    wi_dataset = create_dataset(wi_source, wi_target)
    print("wi_dataset created.")
    
    return [fce_dataset, lang8_dataset, nucle_dataset, wi_dataset] 

# given a list of datasets, tokenize all datasets  
def tokenize_all_datasets(tokenizer, datasets_lst): 
    tokenized_datasets = [] 

    # loop through list of datasets, and tokenize each dataset 
    for dataset in datasets_lst: 
        tokenized_dataset = tokenize_dataset(tokenizer, dataset)
        tokenized_datasets.append(tokenized_dataset)
    
    return tokenized_datasets 
