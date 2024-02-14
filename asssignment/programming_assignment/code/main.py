import numpy as np
import pandas as pd
import os
from platform import python_version
import random
from sklearn.model_selection import train_test_split
# print(python_version())
import torch
from torchtext.data.utils import get_tokenizer
from torch import nn


# Preparing dataset for training & testing
movie_df = pd.read_csv (f'{os.getcwd()}/dataset.csv')
print('Total review',len(movie_df['sentiment']))
print('Pos review',len(movie_df[movie_df['sentiment'] == 'positive'] ))
print('Neg review', len(movie_df[movie_df['sentiment'] == 'negative'] ))

movie_list = movie_df.values.tolist()
random.shuffle(movie_list)
shuffled_df = pd.DataFrame(movie_list, columns=movie_df.columns)
training_dataset, test_dataset = train_test_split(shuffled_df, train_size=40000, test_size=10000)

# Feature extractor: BoW 
# print(training_dataset)

# for i in training_dataset['review']:
    # print(i)
    
word_to_ix = {}
tokenizer = get_tokenizer("basic_english")
tokenized_sentences = [tokenizer(sentence) for sentence in list(movie_df.review)]
print('Length of tokenized sentence', len(tokenized_sentences))

all_tokens = [token for doc in tokenized_sentences for token in doc]
print('Length of tokens', len(all_tokens))
# Create a vocabulary from the tokens
vocab = set(all_tokens)

print('Length of vocab',len( vocab))

# Create a mapping from token to index
word_to_index = {word: i for i, word in enumerate(vocab)}

# Create a bag of words representation
bag_of_words = torch.zeros(len(vocab))
for token in all_tokens:
    bag_of_words[word_to_index[token]] += 1

print("Bag of Words representation:")
print(bag_of_words)

class classifier(nn.Module):
    def __init__(self,x):
        super(self)
        self.x = x

def forward(self,x):
    # wx+b
        logits = self.fc(x)
        return logits
    

def train(model, data_loader, optimizer, scheduler, device):
    model.train()
    for batch in data_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids']
        labels = batch['label']
        outputs = model(input_ids)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        # scheduler.step()
        
        
model = classifier(bag_of_words)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9) 
num_epochs = 20
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    train(model, training_dataset, optimizer,)
    # accuracy, report = evaluate(model, val_dataloader, device)
    # print(f"Validation Accuracy: {accuracy:.4f}")
    # print(report)


    
    

    

