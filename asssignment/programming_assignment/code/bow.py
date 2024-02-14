import numpy as np

text1 = 'nice movie. I like it.'
text2 = 'great movie. I love this.'

tokenized_text1 = text1.split(' ')
tokenized_text2 = text2.split(' ')

corpus = text1 + ' ' + text2
corpus = corpus.replace('.', '')


tokenized_corpus = corpus.split(' ')

corpus_vocab = set(tokenized_corpus)

print(tokenized_corpus)

print("Unique words in the corpus:", corpus_vocab)
count = {}
for word in tokenized_corpus:
    print(word)
    if word in count:
        count[word] +=count[word]
    else:
        count[word] = 1
    
    
print(count)


# FROM OTHER SOURCES
docs = ['nice movie. I like it.',  'great movie. I love this.']

# doing loop vs map is same thing
docs_arr  = []
for doc in docs:
    docs_arr.append(doc.lower())
print(docs_arr)

docs =list(map(str.lower, docs))

print(docs)

unique_words =set((docs[0] + ' ' + docs[1]).split())


# words into number(index)
mapper_word_index = {}
for index,word in enumerate(sorted(unique_words)):
    # print(index, word)
    mapper_word_index[word] = index
     
print(len(mapper_word_index),'length of vocab', mapper_word_index, unique_words)

text1_vector = np.zeros((8)) # For text 1, vector should be => [0,1,1,1,0,1,1,0]
corpus_matrix_embedding = np.zeros((len(docs),len(mapper_word_index)))
print(corpus_matrix_embedding)


for unique_word in mapper_word_index:
    for sentence in docs:
        print(sentence) 
    if unique_word in text1.lower().split():
        print(unique_word, mapper_word_index[unique_word])
        text1_vector[mapper_word_index[unique_word]] = 1

# print('vector representation of text1',text1_vector)
        



for index,sentence in enumerate(docs):
    for unique_word in mapper_word_index:
        # print(sentence)
        if unique_word in sentence:
            # print('here',unique_word, mapper_word_index[unique_word])
            corpus_matrix_embedding[index][mapper_word_index[unique_word]] = 1
            
            
print(corpus_matrix_embedding) # docs = ['nice movie. I like it.',  'great movie. I love this.'] = docs = [[0,1,1,1,0,1,1,0],[1,1,0,0,1,1,0,1]]
    
    


