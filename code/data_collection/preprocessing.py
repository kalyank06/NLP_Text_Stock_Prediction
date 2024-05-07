import json
import pytz
import pandas as pd
import os
from datetime import datetime, timedelta
import datasets
from transformers import AutoTokenizer
from itertools import chain

def load_json_files(directory):
    """ Load all json files from a given directory. """
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            with open(os.path.join(directory, filename), 'r') as file:
                try:
                    yield json.load(file)
                except json.JSONDecodeError:
                    print(f"Error reading {filename}. It's not valid JSON.")

def change_time_to_est(article):
    """ Change the time of the article to EST. """
    time = datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=pytz.UTC)
    article['publishedAt'] = time.astimezone(pytz.timezone('US/Eastern'))
    return article

def process_stock_data(stock_data):
    """ Convert the stock data date to datetime and localize to EST. """
    est = pytz.timezone('US/Eastern')
    stock_data['date'] = stock_data['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').replace(hour=9, minute=0, second=0))
    stock_data['date'] = stock_data['date'].apply(lambda x: est.localize(x))
    return stock_data

def get_articles_for_date(date, all_news, stock_data):
    """ Get articles for a specific date. """
    ind = stock_data[stock_data['date'] == date].index[0]
    if ind != 0:
        start_date = stock_data['date'][ind - 1]
    else:
        start_date = date - timedelta(days=3)
    
    articles = [article for article in all_news if start_date <= article['publishedAt'] <= date]
    return {
        'Date': date,
        'Title': [article['title'] for article in articles],
        'Label': stock_data['price_movement'].loc[ind]
    }

def prepare_dataset(block):
    """ Prepare the dataset from the block. """
    train = pd.DataFrame.from_dict(block['train'])
    test = pd.DataFrame.from_dict(block['test'])
    train['Title'] = train['Title'].apply(lambda x: ' '.join(x))
    test['Title'] = test['Title'].apply(lambda x: ' '.join(x))
    ds_train = datasets.Dataset.from_pandas(train)
    ds_test = datasets.Dataset.from_pandas(test)
    return datasets.DatasetDict({'train': ds_train, 'test': ds_test})

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_function(examples):
    return tokenizer(examples["Title"], padding="max_length", truncation=True)

# Main process
news_directories = ['../../data/news_data/business_news', '../../data/news_data/stock_news']
all_news = list(chain.from_iterable(load_json_files(directory) for directory in news_directories))
all_news = list(map(change_time_to_est, all_news))

stock_data = pd.read_csv('../../data/stock_data/daily_price_movement.csv')
stock_data = process_stock_data(stock_data)

# Process the news and stock data
articles_by_date = [get_articles_for_date(date, all_news, stock_data) for date in stock_data['date']]

def split_into_blocks(data, num_blocks):
    """ Split data into specified number of blocks. """
    block_size = len(data) // num_blocks
    remainder = len(data) % num_blocks
    return [data[i * block_size + min(i, remainder):(i + 1) * block_size + min(i + 1, remainder)] for i in range(num_blocks)]

def train_test_split(block, train_ratio=0.8):
    """ Split each block into training and testing sets. """
    split_index = int(len(block) * train_ratio)
    return {'train': block[:split_index], 'test': block[split_index:]}

# Process the news and stock data into blocks
num_blocks = 5  # Adjust the number of blocks as needed
blocks = split_into_blocks(articles_by_date, num_blocks)

# Split each block into training and testing sets
data_splits = [train_test_split(block) for block in blocks]

# Prepare datasets from each split
processed_data = {f'block{i}': prepare_dataset(split) for i, split in enumerate(data_splits)}

# Save the datasets to disk and tokenize if needed
for block_name, dataset in processed_data.items():
    dataset.save_to_disk(f'../../output/news_data_blocks/{block_name}')
