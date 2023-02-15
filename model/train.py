import argparse
import pandas as pd
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer 
from datasets import load_metric
from utils import * 


def tokenized(data_df):
    X_train_tokenized = tokenizer(data_df['sentence'].to_list(), padding=True, truncation=True, max_length=max_seq_length)
    Y_train = data_df['label'].to_list()
    return X_train_tokenized, Y_train


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='train.py')
    parser.add_argument('--datapath', type=str, default='dataset', help='path to dataset')
    parser.add_argument('--batch', type=int, default=16, help='batch size')
    parser.add_argument('--epoch', type=int, default=5, help='epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--modelpath', type=str, default='xlm-roberta-large', help='path to transformer-based language model')
    parser.add_argument('--modelout', type=str, default='runs/outputs', help='path to save finetuned model')
    parser.add_argument('--savemodel', type=str, default='runs/models', help='path to save finetuned model')
    parser.add_argument('--maxlen', type=int, default=128, help='maximum sequence length')
    opt = parser.parse_args()

    dataset_path = opt.datapath
    model_checkpoint = opt.modelpath 
    batch_size = opt.batch
    epochs = opt.epoch
    lr=opt.lr
    max_seq_length = opt.maxlen
    model_output_path = opt.modelout
    save_model_path = opt.savemodel

    train_path = f'{dataset_path}/train.csv'
    test_path = f'{dataset_path}/test.csv'
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)

    id2label = {
        0: 0, # non-ironic
        1: 1 # ironic
    }
    label2id = {id2label[i]:i for i in range(len(id2label))}
    label_list = list(label2id.keys())
    print(label2id)
    print(id2label) 

    _, dataset_df_train, dataset_df_eval = create_datasets(df_train, label2id, 1)
    _, dataset_df_eval, _ = create_datasets(df_test, label2id, 1)

    X_train_tokenized, Y_train = tokenized(dataset_df_train)
    X_eval_tokenized, Y_eval = tokenized(dataset_df_eval)

    args = TrainingArguments(
        model_output_path,
        evaluation_strategy = "epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        save_strategy = 'epoch',
        load_best_model_at_end=True,
        gradient_accumulation_steps=8,
    )   

    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))

    metric = load_metric("seqeval")

    trainer = Trainer(
        model,
        args,
        train_dataset=Dataset(X_train_tokenized, Y_train),
        eval_dataset=Dataset(X_eval_tokenized, Y_eval),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    trainer.save_model(save_model_path)
