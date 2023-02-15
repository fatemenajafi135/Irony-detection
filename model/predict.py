import argparse
import pandas as pd
import numpy as np
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer 
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, precision_score, f1_score
from datasets import load_metric
from utils import * 

def predict_text(input_text, trainer):
    tokenized_text = tokenizer(input_text, padding=True, truncation=True, max_length=max_seq_length)
    predictions, labels, _ = trainer.predict(Dataset(tokenized_text))
    preds = np.argmax(predictions, axis=-1).tolist()
    preds_tag = [id2label[p] for p in preds]
    return preds_tag, preds

def final_compute_metrics(pred, labels):
    accuracy = accuracy_score(y_true=labels, y_pred=pred, normalize=True)
    recall = recall_score(y_true=labels, y_pred=pred, average='macro')
    precision = precision_score(y_true=labels, y_pred=pred, average='macro')
    f1 = f1_score(y_true=labels, y_pred=pred, average='macro')

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

def tokenized(data_df):
    X_train_tokenized = tokenizer(data_df['sentence'].to_list(), padding=True, truncation=True, max_length=max_seq_length)
    Y_train = data_df['label'].to_list()
    return X_train_tokenized, Y_train


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='predict.py')
    parser.add_argument('--datapath', type=str, default='dataset', help='path to dataset')
    parser.add_argument('--batch', type=int, default=16, help='batch size')
    parser.add_argument('--epoch', type=int, default=5, help='epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--modelpath', type=str, default='xlm-roberta-large', help='path to transformer-based language model')
    parser.add_argument('--maxlen', type=int, default=128, help='maximum sequence length')
    parser.add_argument('--predspath', type=str, default='runs/preds/', help='path for preditions of test set')
    opt = parser.parse_args()

    dataset_path = opt.datapath
    model_checkpoint = opt.modelpath 
    predictions_path = opt.predspath
    batch_size = opt.batch
    epochs = opt.epoch
    lr=opt.lr
    max_seq_length = opt.maxlen

    test_path = f'{dataset_path}/test.csv'
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

    _, dataset_test, _ = create_datasets(df_test, label2id, 1)

    X_test, Y_test = tokenized(dataset_test)
    X_train, Y_train = X_test, Y_test

    args = TrainingArguments(
        'runs/temp_output',
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
        train_dataset=Dataset(X_test, Y_test),
        eval_dataset=Dataset(X_test, Y_test),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    preds, _ = predict_text(list(df_test['tweet']), trainer) 

    predicts = pd.DataFrame({
        'tweet': df_test['tweet'],
        'label': df_test['label'],
        'preds': preds
    })

    predicts.to_csv(f'{predictions_path}/preds.csv', index=False)

    print(final_compute_metrics(preds, list(df_test['label'])))