import os
import json
import pandas as pd

PATH = './crawled_messages'

def wanted_info(post):
    info = {'message': post['message']}
    if 'reactions' in post.keys():
        for react in post['reactions']['results']:
            info[react['reaction']['emoticon']] = react['count']
    info['date'] = post['date']
    info['media'] = 1 if 'media' in post.keys() else 0
    return info

def is_unique(message):
    if message in uniques: 
        return False
    uniques.add(message)
    return True


if __name__ == '__main__':

    uniques = set()
    messages = []

    if os.path.exists(PATH):

        files = sorted([f'{PATH}/{file_name}' for file_name in os.listdir(PATH)], reverse=True) 
        
        for file_name in files:
            
            print(file_name)
            with open (file_name) as f:
                data = json.load(f)

            posts = [wanted_info(post) for post in data['messages']]
            posts = [post for post in posts if is_unique(post['message'])]
        
            messages += posts

        messages_df = pd.DataFrame(messages)
        messages_df.to_csv('messages.csv', index=False)
        print(messages_df.shape)
    
    else:
        print("Path doesn't exist:", PATH)