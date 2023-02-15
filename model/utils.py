import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, precision_score, f1_score


class Dataset(torch.utils.data.Dataset):
    
  def __init__(self, encodings, labels=None):
        self.encodings = encodings        
        self.labels = labels

  def __len__(self):
        return len(self.encodings["input_ids"])

  def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

def create_datasets(data, lable2id, train_test_split_ratio=1):
  
    sentences = []
    labels = []

    for idx, row in data.iterrows():
        # print(row)
        if row['tweet'] == 'Tweet': continue
        sent = row['tweet']
        tag_num = lable2id[row['label']]

        if pd.isnull(sent): continue

        sentences.append(sent)
        labels.append(tag_num)

    ##### Converting lists to dataframes
    dataset_df = pd.DataFrame()
    dataset_df["sentence"] = sentences
    dataset_df["label"] = labels

    ##### Making Dataset: with splitting - for each class
    dataset_df_train = pd.DataFrame()
    dataset_df_test = pd.DataFrame()

    for k in lable2id.keys():
        train_test_split_point = int(train_test_split_ratio*len(dataset_df[dataset_df['label'] == lable2id[k]]))
        train_new_part = dataset_df[dataset_df['label'] == lable2id[k]][:train_test_split_point]
        dataset_df_train = pd.concat([dataset_df_train, train_new_part])
        test_new_part = dataset_df[dataset_df['label'] == lable2id[k]][train_test_split_point:]
        dataset_df_test = pd.concat([dataset_df_test, test_new_part])
    # print('train and test sizes: ', dataset_df_train.shape, dataset_df_test.shape)
    return lable2id, dataset_df_train, dataset_df_test



def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred, normalize=True)
    # accuracy = balanced_accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred, average='macro')
    precision = precision_score(y_true=labels, y_pred=pred, average='macro')
    f1 = f1_score(y_true=labels, y_pred=pred, average='macro')

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}