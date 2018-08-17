import pandas as pd
import numpy as np 
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
from math import ceil
import random
import pickle
def preprocess_docs():
  #load data from file
  train_df = pd.read_csv('data/train.csv')
  raw_docs_train = train_df['comment_text'].fillna("_na_").values #replace missing values N/A with '_na_'
  class_list = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
  label_train = train_df[class_list].values

  test_df = pd.read_csv('data/test.csv')
  raw_docs_test = test_df['comment_text'].fillna("_na_").values

  num_labels = len(class_list)

  #tokenize the comment and preprocess the text
  print ("start to preprocess training data...")
  stop_words = set(stopwords.words('english'))
  stop_words.update(['[', ']', '*', '{', '}', '(', ')', '"', "'", ':', '|', '/', '\/', ':', '#', '=', '~', '†', 'با', '-', '_'])
  print ("stop_words: ", stop_words)
  
  processed_docs_train = []
  counter = Counter()
  for doc in raw_docs_train:
    doc = doc.lower()
    doc = doc.replace(',', ' , ')
    doc = doc.replace('!', ' ! ')
    doc = doc.replace('.', ' . ')
    doc = doc.replace('?', ' ? ')
    doc = doc.replace(';', ' ; ')
    doc = doc.replace('-', ' - ')
    doc = doc.replace('_', ' _ ')
    doc = doc.replace('"', ' " ')
    doc = doc.replace('(', ' ( ')
    doc = doc.replace("'", " ' ")
    doc = doc.replace(':', ' : ')
    doc = doc.replace('=', ' = ')
    doc = doc.replace('/', ' / ')
    doc = doc.replace('|', ' | ')
    doc = doc.replace('}', ' } ')
    doc = doc.replace('{', ' { ')
    doc = doc.replace('(', ' ( ')
    doc = doc.replace(')', ' ) ')
    doc = doc.replace('[', ' [ ')
    doc = doc.replace(']', ' ] ')
    doc = doc.replace('<', ' < ')
    doc = doc.replace('&', ' & ')
    doc = doc.replace('*', ' * ')
    doc = doc.replace('#', ' # ')
    doc = doc.replace('\/', ' \/ ')
    doc = doc.split()
    lemm = WordNetLemmatizer()
    lemmed = [lemm.lemmatize(word) for word in doc if word not in stop_words]
    processed_docs_train.append(lemmed)
    counter.update(lemmed)

  print ("start to preprocess test data...")
  processed_docs_test = []
  for doc in raw_docs_test:
    doc = doc.lower()
    doc = doc.lower()
    doc = doc.replace(',', ' , ')
    doc = doc.replace('!', ' ! ')
    doc = doc.replace('.', ' . ')
    doc = doc.replace('?', ' ? ')
    doc = doc.replace(';', ' ; ')
    doc = doc.replace('-', ' - ')
    doc = doc.replace('_', ' _ ')
    doc = doc.replace('"', ' " ')
    doc = doc.replace('(', ' ( ')
    doc = doc.replace("'", " ' ")
    doc = doc.replace(':', ' : ')
    doc = doc.replace('=', ' = ')
    doc = doc.replace('/', ' / ')
    doc = doc.replace('|', ' | ')
    doc = doc.replace('}', ' } ')
    doc = doc.replace('{', ' { ')
    doc = doc.replace('(', ' ( ')
    doc = doc.replace(')', ' ) ')
    doc = doc.replace('[', ' [ ')
    doc = doc.replace(']', ' ] ')
    doc = doc.replace('<', ' < ')
    doc = doc.replace('&', ' & ')
    doc = doc.replace('*', ' * ')
    doc = doc.replace('#', ' # ')
    doc = doc.replace('\/', ' \/ ')

    doc = doc.split()
    lemm = WordNetLemmatizer()
    lemmed = [lemm.lemmatize(word) for word in doc if word not in stop_words]
    processed_docs_test.append(lemmed)
    counter.update(lemmed)

  print ("in total %s samples in train.csv" %len(processed_docs_train))
  print ("in total %s samples in test.csv" %len(processed_docs_test))

  print('%s words found in all docs' %len(counter))
  print ("build the vocabulary...")
  min_word_count = 100
  vocab = [word for word,freq in counter.items() if freq>min_word_count]
  print ("%s words in final vocabulary" %len(vocab))

  word2ind = {word:(ind+2) for ind, word in enumerate(vocab)}

  UNKNOWN_CHAR = "UNK"
  PAD_CHAR = "PAD"
  word2ind[UNKNOWN_CHAR] = 0
  word2ind[PAD_CHAR] = 1
  vocab_size = len(word2ind)
  ind2word = {ind:word for word, ind in word2ind.items()}
  with open("word2ind.txt", 'wb')  as file:
    file.write(pickle.dumps(word2ind))
  with open("ind2word.txt", 'wb')  as file:
    file.write(pickle.dumps(ind2word))
  
  glove_dict = {}
  print ("loading GLove model...")
  f = open("../glove.6B/glove.6B.100d.txt", 'r', encoding="utf8")
  for line in f:
    splitline = line.split()
    word = splitline[0]
    word_vector =np.array([float(val) for val in splitline[1:]])
    glove_dict[word] = word_vector

  all_embs = np.stack(glove_dict.values())
  emb_mean,emb_std = all_embs.mean(), all_embs.std()

  word_embeddings = np.zeros((vocab_size, 100))
  for word, id in word2ind.items():
    word_vector = glove_dict.get(word)
    if word_vector is not None:
      word_embeddings[id] = word_vector
    else:
      word_embeddings[id] = np.random.normal(emb_mean, emb_std, 100)


  print ("convert words to index in vocabulary")
  word_id_train = []
  for doc in processed_docs_train:
    doc_ids = [word2ind.get(word, word2ind[UNKNOWN_CHAR]) for word in doc]
    word_id_train.append(doc_ids)

  word_id_test = []
  for doc in processed_docs_test:
    doc_ids = [word2ind.get(word, word2ind[UNKNOWN_CHAR]) for word in doc]
    word_id_test.append(doc_ids)



  print ("split the train.csv into training set and validation set...")
  num_training_samples = int(len(word_id_train)*0.9)
  word_id_val = word_id_train[num_training_samples:]
  word_id_train = word_id_train[:num_training_samples]
  label_val = label_train[num_training_samples:]
  label_train = label_train[:num_training_samples]

  print ("using %d samples for training and %d samples for validation" % (len(word_id_train), len(word_id_val)))

  np.save("word_id_train.npy", word_id_train)
  np.save("word_id_val.npy", word_id_val)
  np.save("label_train.npy", label_train)
  np.save("label_val.npy", label_val)
  np.save("word_id_test.npy", word_id_test)
  np.save("word_embeddings.npy", word_embeddings)

def shuffle_data(data, label):
  num = len(data)
  ids = list(range(num))
  random.shuffle(ids) #shuffle will change the list in-place
  return [data[i] for i in ids], [label[i] for i in ids]


def generate_batches(data, label, word2id, batch_size):

  num_batches = ceil(len(data)/float(batch_size))
  data_batches=[]
  label_batches = []
  length_batches =[]
  for batch in range(num_batches):
    start = batch_size*batch
    end = min(batch_size*(batch+1), (len(data)))
    if label != []:
      label_batches.append(label[start:end])
    else:
      label_batches = []
    lengths = [len(data[i]) for i in range(start, end)]
    
    max_length = min(200, max(lengths))
    data_padded = []
    length_padded = []
    for i in data[start:end]:
      if len(i)<max_length:
        length_padded.append(len(i))
        [i.append(word2id['PAD']) for p in range(max_length-len(i))]

      else:
        length_padded.append(max_length)
        i = i[:max_length]
      data_padded.append(i)

    data_batches.append(data_padded)
    length_batches.append(length_padded)


  return data_batches, label_batches, length_batches
    
if __name__ == "__main__":
  preprocess_docs()

