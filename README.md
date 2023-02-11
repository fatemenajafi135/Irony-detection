# Persian Irony Detection using transformer-based language models

- input: a text in Persian
- output: classifying text as ironic and non-ironic

## Dataset

**Existing datasets**: 
- Persian manually labeled dataset: [MirasIrony](https://github.com/miras-tech/MirasText/tree/master/MirasIrony) 

- Persian automatically labeled dataset: [Persian Irony Detection](https://github.com/fatemenajafi135/Irony-detection/dataset)


**Create new dataset steps** (Crawling Persian tweets from a channel on Telegram and automatically labeling them)
- **crawling**: Crawl public channels' messages on Telegram using the api of [Telegram server](https://tg.i-c-a.su/) in file ```crawling.py```. Save crawled messages (json files) in ```./crawled_messages``` 
- **gathering**: Concatenate crawled files, save wanted attributes of each tweet in a Pandas DataFrame, and save it in a csv file. The file ```gathering.py``` creates ```messages.csv```. 
- **cleaning**: Basic clean on the previously created dataset and save it to ```messages_cleaned.csv``` 
- **labeling**: Set label to each tweet by its top-2 common reactions and split dataset to Train and Test sets. It saves files in ```../dataset/```. 

- **Run**: (The previous dataset will be replaced)
```
cd creating_dataset/
pip install requirements.txt
python crawling.py
python gathering.py
python cleaning.py
python labeling.py
```

## Model
Finetuning an uncased language model on the Persian irony detection dataset