import pandas as pd
from datetime import datetime

def convert_date(timestamp):
  dt_object = datetime.fromtimestamp(timestamp)
  return dt_object.strftime('%Y-%m-%d')

def basic_clean(text):
    text = text.split('<br />\n')
    text = ' '.join(text[:-4])
    text = ' '.join(text.split())
    return text

def is_typical(text):
    text = str(text).split('<br />\n')
    last_line = '<a href="tg://resolve?domain=@OfficialPersianTwitter" rel="nofollow">@OfficialPersianTwitter</a>'
    return text[-1] == last_line

if __name__ == '__main__':

    df = pd.read_csv('messages.csv')
    print(df.columns)
    df = df[df.media == 0]
    df = df[df['message'].apply(lambda message: is_typical(message))]
    df['message'] = df['message'].apply(lambda message: basic_clean(message))
    df['date'] = df['date'].apply(lambda time: convert_date(time)) 
    
    emoji_mapper = {'❤': 'heart', '👍': 'like', '👎': 'dislike', '😁': 'laugh', '😢': 'cry'}
    for e in emoji_mapper:
        df[e] = df[e].fillna(0)
    df = df.rename(columns=emoji_mapper)
    
    df = df[['message', 'date', 'like', 'dislike', 'laugh', 'cry', 'heart']]
    df.to_csv('messages_cleaned.csv', index=False)
    print(df.shape)
    print(df.columns)
