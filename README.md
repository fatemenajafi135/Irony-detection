# Persian Irony Detection using transformer-based language models

- input: a text in Persian
- output: classifing text as ironic and non-ironic

## Dataset
**Existing datasets**: Persian Manually labeled dataset: [MirasIrony](https://github.com/miras-tech/MirasText/tree/master/MirasIrony) 

**Create new dataset**: Crawling Persian tweets from a channel on Telegram and automaticaly labeling them
- **crawling**: crawled public channels' messeges on Telegram using [Telegram server](https://tg.i-c-a.su/)
- **gathering**: 
- **cleaning and labeling**:

## Model
Finetuning an uncased language model on the Persian irony detection dataset