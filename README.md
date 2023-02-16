# Persian Irony Detection using transformer-based language models

- input: a text in Persian
- output: classifying text as ironic and non-ironic

## Dataset

**Existing datasets**: 
- Persian manually labeled dataset: [MirasIrony](https://github.com/miras-tech/MirasText/tree/master/MirasIrony) 

- Persian automatically labeled dataset: [Persian Irony Detection](https://github.com/fatemenajafi135/Irony-detection/tree/main/dataset)


**Create new dataset steps** (Crawling Persian tweets from a channel on Telegram and automatically labeling them)
- **crawling**: Crawl public channels' messages on Telegram using the api of [Telegram server](https://tg.i-c-a.su/) in file ```crawling.py```. Save crawled messages (json files) in ```./crawled_messages``` 
- **gathering**: Concatenate crawled files, save wanted attributes of each tweet in a Pandas DataFrame, and save it in a csv file. The file ```gathering.py``` creates ```messages.csv```. 
- **cleaning**: Basic clean on the previously created dataset and save it to ```messages_cleaned.csv``` 
- **labeling**: Set label to each tweet by its top-2 common reactions and split dataset to Train and Test sets. It saves files in ```../dataset/```. 

- **Run**: (The previous dataset will be replaced)
``` shell
cd creating_dataset/
pip install requirements.txt
python crawling.py
python gathering.py
python cleaning.py
python labeling.py
```

## Model
Finetuning an uncased language model on the Persian irony detection dataset

``` shell
cd model/ 
pip install -r requirements.txt
```

**Finetuning** a transformer-based language model on irony detection dataset

``` shell
python train.py  --datapath [path to dataset] --modelpath [path to transformer-based language model] --modelout [path to save finetuned model] --savemodel [path to save finetuned model] --maxlen [maximum sequence length] --batch [batch size] --epoch [epochs] --lr [learning rate]
# example
python train.py --datapath ../dataset/ --modelpath xlm-roberta-base --batch 16 --epoch 5
```

**Predict** label using trained model

``` shell
python predict.py  --datapath [path to dataset] --modelpath [path to transformer-based language model] --predspath [path for preditions of test set] --maxlen [maximum sequence length] --batch [batch size] --epoch [epochs] --lr [learning rate]
# example
python predict.py --datapath ../dataset/ --modelpath xlm-roberta-base --predspath runs/preds
```

## Results

Comparison of different finetuned language models on the Persian dataset   

| Language Model | Accuracy | Recall | Precision | F1 |
| - | - | - | - | - |
| [ParsBert vr3](https://huggingface.co/HooshvareLab/bert-fa-zwnj-base) | 81.3% | 81.4% | 81.3% | 81.3% |
| [XLM-RoBERTa-Base](https://huggingface.co/xlm-roberta-base) | 82.6% | 82.8% | 82.6% | 82.5% |
| [__XLM-RoBERTa-Large__](https://huggingface.co/xlm-roberta-large) | __84.7%__ | __84.7%__ | __84.6%__ | __84.6%__ |

