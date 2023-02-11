import os 
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split

PATH = '../dataset/'

def sort_emojis(info):
    reacts = Counter({reaction: info[reaction] for reaction in reactions})
    return [common[0] for common in reacts.most_common(2)]

def tagger(row):
    emoji_1, emoji_2 = sort_emojis(row)
    if emoji_1 == 'laugh':
        return 1
    if emoji_1 == 'like' and emoji_2 == 'laugh':
        return -1
    return 0


if __name__ == '__main__':

    reactions = ['like', 'dislike', 'laugh', 'cry', 'heart']

    if not os.path.exists(PATH):
        os.makedirs(PATH)


    df = pd.read_csv('messages_cleaned.csv')
    df['label'] = df.apply(lambda row: tagger(row), axis=1)
    df = df.rename(columns={'message': 'tweet'})
    df = df[['tweet', 'label']]
    df.to_csv(f'{PATH}Persian_irony_detection.csv', index=False)
    print(f'Persian irony detection size: {df.shape}')

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=135)
    print(f'train size: {df_train.shape}, test size: {df_test.shape}')

    df_train.to_csv(f'{PATH}train.csv', index=False)
    df_test.to_csv(f'{PATH}test.csv', index=False)