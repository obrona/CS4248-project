# requirements: transformers
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

def train_model(tokenizer, model, output_dir, tokenized_dataset): 
    # prepare data as batches 
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # define training settings  
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        predict_with_generate=True,
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none"
    )

    # create trainer 
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        processing_class=tokenizer,
        data_collator=data_collator
    )

    # train model 
    trainer.train()

    # save model and tokenizer 
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("training completed. model and tokenizer saved in: ", output_dir)  

# train model on multiple datasets 
def train_model_alldatasets(tokenizer, model, output_dir, tokenized_dataset_lst): 
    # loop through list of tokenized datasets 
    for tokenized_dataset in tokenized_dataset_lst: 
        train_model(tokenizer, model, output_dir, tokenized_dataset) 

    print("training completed for all datasets")