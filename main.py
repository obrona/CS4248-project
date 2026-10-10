import model_init as model_init 
import data_preprocessing as data_preprocessing 
import model_training as model_training 

OUTPUT_DIR = r""

def main(): 
    # initialise tokenizer and model 
    tokenizer, model = model_init.initialise_small_model() 

    # create and tokenize datasets 
    dataset_lst = data_preprocessing.create_all_datasets() 
    tokenized_dataset_lst = data_preprocessing.tokenize_all_datasets(tokenizer, dataset_lst) 

    # train test split 

    # train model 
    model_training.train_model_alldatasets(tokenizer, model, OUTPUT_DIR, tokenized_dataset_lst) 

    # temporary 
    print("main completed.")