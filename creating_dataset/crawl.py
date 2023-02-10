import requests
import json
import time
import os

CHANNEL_NAME = 'OfficialPersianTwitter'
PATH = './crawled_messages'
COUNT = 100

def call_telegram_api(addr):
    resp = requests.get(url=addr, params={})
    return resp.json()


if __name__ == '__main__':

    if not os.path.exists(PATH):
        os.makedirs(PATH)

    print('Crawling started!')

    for i in range(COUNT):
        print(i, '--------------------------------------------')
        messages = []
        for j in range(10):
            ts = int(time.time())
            page = i * 10 + j + 1
            recent_messages = call_telegram_api(f'https://tg.i-c-a.su/json/{CHANNEL_NAME}/{page}?limit=100')['messages']
            print(page, ts)
            messages += recent_messages        
            
        messages = {'messages': messages}
        with open(f'{PATH}/{ts}-{i}.json', 'w') as f:
            json.dump(messages, f)
        
        wait = 60   # for reguest limit 
        print(f'Zzz! {wait} sec')
        time.sleep(wait)