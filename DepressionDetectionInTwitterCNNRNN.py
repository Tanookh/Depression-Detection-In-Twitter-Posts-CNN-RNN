# %%
"""
# Depression Detection in Social Media Posts

#### Imports
"""

# %%
import warnings
warnings.filterwarnings("ignore")
import ftfy
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import re

from math import exp
from numpy import sign

from sklearn.metrics import  classification_report, confusion_matrix, accuracy_score
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk import PorterStemmer

from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv1D, Dense, Input, LSTM, Embedding, Dropout, Activation, MaxPooling1D, Flatten, GlobalMaxPooling1D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing import text
from keras.utils import np_utils
from keras.preprocessing import sequence

# %%
"""
#### Constants
"""

# %%
# Reproducibility
np.random.seed(1234)

DEPRES_NROWS = 3200  # number of rows to read from DEPRESSIVE_TWEETS_CSV
RANDOM_NROWS = 12000 # number of rows to read from RANDOM_TWEETS_CSV
MAX_SEQUENCE_LENGTH = 140 # Max tweet size
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 300
TRAIN_SPLIT = 0.6
TEST_SPLIT = 0.2
EPOCHS= 3

# %%
"""
## Section 1: Load Data

Loading depressive tweets scraped from twitter using [TWINT](https://github.com/haccer/twint) and random tweets from Kaggle dataset [twitter_sentiment](https://www.kaggle.com/ywang311/twitter-sentiment/data).

#### File Paths
"""

# %%
#DEPRESSIVE_TWEETS_CSV = 'depressive_tweets.csv'
DEPRESSIVE_TWEETS_CSV = 'depressive_tweets_processed.csv'
RANDOM_TWEETS_CSV = 'Sentiment Analysis Dataset 2.csv'
EMBEDDING_FILE = 'GoogleNews-vectors-negative300.bin'
 ## a pre-trained word2vec model by google for sentiment analysis.

# %%
depressive_tweets_df = pd.read_csv(DEPRESSIVE_TWEETS_CSV, sep = '|',
                                   header = None, usecols = range(0,9), nrows = DEPRES_NROWS)
random_tweets_df = pd.read_csv(RANDOM_TWEETS_CSV, encoding = "ISO-8859-1",
                               usecols = range(0,4), nrows = RANDOM_NROWS)

# %%
depressive_tweets_df.head()

# %%
random_tweets_df.head()

# %%
"""
## Section 2: Data Processing
"""

# %%
"""
### Load Pretrained Word2Vec Model

The pretrained vectors for the Word2Vec model is from [here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit).
Using a Keyed Vectors file, we can get the embedding of any word by calling `.word_vec(word)` and we can get all the words in the model's vocabulary through `.vocab`.
"""

# %%
word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)

# %%
"""
### Preprocessing

Preprocessing the tweets in order to:
* Remove links and images
* Remove hashtags
* Remove @ mentions
* Remove emojis
* Remove stop words excpet first, second and third pronouns
* Get rid of stuff like "what's" and making it "what is'
* Stem words so they are all the same tense (e.g. ran -> run)
"""

# %%
# Expand Contraction
cList = {
  "ain't": "am not",
  "aren't": "are not",
  "can't": "cannot",
  "can't've": "cannot have",
  "'cause": "because",
  "cause": "because",
  "could've": "could have",
  "couldn't": "could not",
  "couldn't've": "could not have",
  "didn't": "did not",
  "doesn't": "does not",
  "don't": "do not",
  "hadn't": "had not",
  "hadn't've": "had not have",
  "hasn't": "has not",
  "haven't": "have not",
  "he'd": "he would",
  "he'd've": "he would have",
  "he'll": "he will",
  "he'll've": "he will have",
  "he's": "he is",
  "how'd": "how did",
  "how'd'y": "how do you",
  "how'll": "how will",
  "how's": "how is",
  "I'd": "I would",
  "I'd've": "I would have",
  "I'll": "I will",
  "I'll've": "I will have",
  "I'm": "I am",
  "I've": "I have",
  "isn't": "is not",
  "it'd": "it had",
  "it'd've": "it would have",
  "it'll": "it will",
  "it'll've": "it will have",
  "it's": "it is",
  "let's": "let us",
  "ma'am": "madam",
  "mayn't": "may not",
  "might've": "might have",
  "mightn't": "might not",
  "mightn't've": "might not have",
  "must've": "must have",
  "mustn't": "must not",
  "mustn't've": "must not have",
  "needn't": "need not",
  "needn't've": "need not have",
  "o'clock": "of the clock",
  "oughtn't": "ought not",
  "oughtn't've": "ought not have",
  "shan't": "shall not",
  "sha'n't": "shall not",
  "shan't've": "shall not have",
  "she'd": "she would",
  "she'd've": "she would have",
  "she'll": "she will",
  "she'll've": "she will have",
  "she's": "she is",
  "should've": "should have",
  "shouldn't": "should not",
  "shouldn't've": "should not have",
  "so've": "so have",
  "so's": "so is",
  "that'd": "that would",
  "that'd've": "that would have",
  "that's": "that is",
  "there'd": "there had",
  "there'd've": "there would have",
  "there's": "there is",
  "they'd": "they would",
  "they'd've": "they would have",
  "they'll": "they will",
  "they'll've": "they will have",
  "they're": "they are",
  "they've": "they have",
  "to've": "to have",
  "wasn't": "was not",
  "we'd": "we had",
  "we'd've": "we would have",
  "we'll": "we will",
  "we'll've": "we will have",
  "we're": "we are",
  "we've": "we have",
  "weren't": "were not",
  "what'll": "what will",
  "what'll've": "what will have",
  "what're": "what are",
  "what's": "what is",
  "what've": "what have",
  "when's": "when is",
  "when've": "when have",
  "where'd": "where did",
  "where's": "where is",
  "where've": "where have",
  "who'll": "who will",
  "who'll've": "who will have",
  "who's": "who is",
  "who've": "who have",
  "why's": "why is",
  "why've": "why have",
  "will've": "will have",
  "won't": "will not",
  "won't've": "will not have",
  "would've": "would have",
  "wouldn't": "would not",
  "wouldn't've": "would not have",
  "y'all": "you all",
  "y'alls": "you alls",
  "y'all'd": "you all would",
  "y'all'd've": "you all would have",
  "y'all're": "you all are",
  "y'all've": "you all have",
  "you'd": "you had",
  "you'd've": "you would have",
  "you'll": "you you will",
  "you'll've": "you you will have",
  "you're": "you are",
  "you've": "you have"
}

c_re = re.compile('(%s)' % '|'.join(cList.keys()))

def expandContractions(text, c_re=c_re):
    def replace(match):
        return cList[match.group(0)]
    return c_re.sub(replace, text)

# %%
def clean_tweets(tweets):
    cleaned_tweets = []
    for tweet in tweets:
        tweet = str(tweet)
        # if url links then dont append to avoid news articles
        # also check tweet length, save those > 10 (length of word "depression")
        if re.match("(\w+:\/\/\S+)", tweet) == None and len(tweet) > 10:
            #remove hashtag, @mention, emoji and image URLs
            tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|(\#[A-Za-z0-9]+)|(<Emoji:.*>)|(pic\.twitter\.com\/.*)",
                                    " ", tweet).split())
            
            #fix weirdly encoded texts
            tweet = ftfy.fix_text(tweet)
            
            #expand contraction
            tweet = expandContractions(tweet)

            #stop words
            stop_words = set(stopwords.words('english'))
            
            #####################################################3
            stop_words_remove = ("I","i","me","Me","my","My","mine","Mine","you",
                                 "your","yours","he","she","me","It","him","her",
                                 "his","its","hers","ours","our","we","us","theirs",
                                 "their","them","they","You","Your","Yours","He","She",
                                 "Me","It","Him","Her","His","Its","Hers","Ours","Our",
                                 "We","Us","Theirs","Their","Them","They")
            stop_words_remove = set(stop_words_remove)
            ###########################################################3
            stop_words = stop_words  - stop_words_remove
            word_tokens = nltk.word_tokenize(tweet) 
            filtered_sentence = [w for w in word_tokens if not w in stop_words]
            tweet = ' '.join(filtered_sentence)

            #stemming words
            tweet = PorterStemmer().stem(tweet)
            
            cleaned_tweets.append(tweet)

    return cleaned_tweets

# %%
"""
Applying the preprocessing `clean_text` function to every element in the depressive tweets and random tweets data.
"""

# %%
depressive_tweets_arr = [x for x in depressive_tweets_df[5]]
random_tweets_arr = [x for x in random_tweets_df['SentimentText']]
X_d = clean_tweets(depressive_tweets_arr)
X_r = clean_tweets(random_tweets_arr)

# %%
"""
### Tokenizer

Using a Tokenizer to assign indices and filtering out unfrequent words. Tokenizer creates a map of every unique word and an assigned index to it. The parameter called num_words indicates that we only care about the top 20000 most frequent words.
"""

# %%
tokenizer = Tokenizer(num_words=MAX_NB_WORDS,filters='#$%&()*+-<=>@[\\]^_`{|}~\t\n',
                      lower = False, split = ' ')
tokenizer.fit_on_texts(X_d + X_r)

# %%
"""
Applying the tokenizer to depressive tweets and random tweets data.
"""

# %%
sequences_d = tokenizer.texts_to_sequences(X_d)
sequences_r = tokenizer.texts_to_sequences(X_r)

# %%
"""
Number of unique words in tokenizer. Has to be <= 20,000.
"""

# %%
word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

# %%
"""
Pad sequences all to the same length of 140 words.
"""

# %%
data_d = pad_sequences(sequences_d, maxlen=MAX_SEQUENCE_LENGTH)
data_r = pad_sequences(sequences_r, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data_d tensor:', data_d.shape)
print('Shape of data_r tensor:', data_r.shape)

# %%
"""
### Embedding Matrix

The embedding matrix is a `n x m` matrix where `n` is the
 number of words and `m` is the dimension of the embedding.
 In this case, `m=300`.
 n = (the min (between the number of unique words in our
               tokenizer and the max we specified)).
"""

# %%
nb_words = min(MAX_NB_WORDS, len(word_index))

embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))

for (word, idx) in word_index.items():
    if word in word2vec.vocab and idx < MAX_NB_WORDS:
        embedding_matrix[idx] = word2vec.word_vec(word)

# %%
"""
### Splitting and Formatting Data

Assigning labels to the depressive tweets and random tweets data, and splitting the arrays into test (60%), validation (20%), and train data (20%). Combine depressive tweets and random tweets arrays and shuffle.
"""

# %%
# Assigning labels to the depressive tweets and random tweets data
labels_d = np.array([1] * DEPRES_NROWS)
labels_r = np.array([0] * RANDOM_NROWS)

# Splitting the arrays into test (20%), validation (20%), and train data (60%)
perm_d = np.random.permutation(len(data_d))
idx_train_d = perm_d[:int(len(data_d)*(TRAIN_SPLIT))]
idx_test_d = perm_d[int(len(data_d)*(TRAIN_SPLIT)):int(len(data_d)*(TRAIN_SPLIT+TEST_SPLIT))]
idx_val_d = perm_d[int(len(data_d)*(TRAIN_SPLIT+TEST_SPLIT)):]

perm_r = np.random.permutation(len(data_r))
idx_train_r = perm_r[:int(len(data_r)*(TRAIN_SPLIT))]
idx_test_r = perm_r[int(len(data_r)*(TRAIN_SPLIT)):int(len(data_r)*(TRAIN_SPLIT+TEST_SPLIT))]
idx_val_r = perm_r[int(len(data_r)*(TRAIN_SPLIT+TEST_SPLIT)):]

# Combine depressive tweets and random tweets arrays
data_train = np.concatenate((data_d[idx_train_d], data_r[idx_train_r]))
labels_train = np.concatenate((labels_d[idx_train_d], labels_r[idx_train_r]))
data_test = np.concatenate((data_d[idx_test_d], data_r[idx_test_r]))
labels_test = np.concatenate((labels_d[idx_test_d], labels_r[idx_test_r]))
data_val = np.concatenate((data_d[idx_val_d], data_r[idx_val_r]))
labels_val = np.concatenate((labels_d[idx_val_d], labels_r[idx_val_r]))

# Shuffling
perm_train = np.random.permutation(len(data_train))
data_train = data_train[perm_train]
labels_train = labels_train[perm_train]
perm_test = np.random.permutation(len(data_test))
data_test = data_test[perm_test]
labels_test = labels_test[perm_test]
perm_val = np.random.permutation(len(data_val))
data_val = data_val[perm_val]
labels_val = labels_val[perm_val]

# %%
"""
## Section 3: Building the Model
"""

# %%
"""
### Building Model (CNN With Max)

The model takes in an input and then outputs a single number 
representing the probability that the tweet indicates depression.

The model takes in each input sentence, replace it with it's embeddings, 
then run the new embedding vector through a convolutional layer.

CNNs are excellent at learning spatial structure from data, 
the convolutional layer takes advantage of that and learn some structure 
from the sequential data then #####pass into a standard LSTM layer.

#####Last but not least, the output of the LSTM layer is fed into a 
standard Dense model for prediction.
"""

#######################################################
print("---------CNN With Max starts here---------")
#######################################################

# %%
# a class to build our layers
model = Sequential()
# Embedded layer - to learn the embedding matrix words
model.add(Embedding(len(embedding_matrix), EMBEDDING_DIM,
                    weights=[embedding_matrix], 
                            input_length=MAX_SEQUENCE_LENGTH
                            ,trainable=False))
# dropsout random neurons (ignores 20% of neurons)
model.add(Dropout(0.2))
# Convolutional Layer
model.add(Conv1D(filters=250, kernel_size=3, padding='same', activation='relu'))
model.add(GlobalMaxPooling1D(data_format ='channels_last'))
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# %%
"""
### Compiling Model
"""

# %%
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
print(model.summary())


# %%
"""
## Section 4: Training the Model

The model is trained `EPOCHS` time, and Early Stopping argument is used to end training if the loss or accuracy don't improve within 3 epochs.
"""

# %%
early_stop1 = EarlyStopping(monitor='val_loss', patience=3)

data_train1 = data_train
labels_train1 = labels_train
data_val1 = data_val
labels_val1 = labels_val

hist1 = model.fit(data_train1, labels_train1, \
        validation_data=(data_val1, labels_val1), \
        epochs=EPOCHS, batch_size=32, shuffle=True, \
        callbacks=[early_stop1])

# %%
"""
### Results
"""

# %%
"""
Summarize history for accuracy
"""

# %%
plt.plot(hist1.history['acc'])
plt.plot(hist1.history['val_acc'])
plt.title('CNN With Max model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# %%
"""
Summarize history for loss
"""

# %%
plt.plot(hist1.history['loss'])
plt.plot(hist1.history['val_loss'])
plt.title('CNN With Max model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# %%
"""
Percentage accuracy of model
"""

# %%
data_test1 = data_test
labels_test1 = labels_test
labels_pred1 = model.predict(data_test1)
labels_pred1 = np.round(labels_pred1.flatten())
accuracy1 = accuracy_score(labels_test1, labels_pred1)
print("Accuracy: %.2f%%" % (accuracy1*100))

# %%
"""
f1, precision, and recall scores
"""

# %%
print(classification_report(labels_test1, labels_pred1))

#######################################################
print("---------CNN With Max ends here---------")
#######################################################


from keras import backend as K 
import gc
K.clear_session()
gc.collect()


#######################################################
print("---------Multi Channel CNN starts here---------")
#######################################################

# %%
# a class to build our layers
model = Sequential()
# Embedded layer - to learn the embedding matrix words
model.add(Embedding(len(embedding_matrix), EMBEDDING_DIM,
                    weights=[embedding_matrix], 
                            input_length=MAX_SEQUENCE_LENGTH
                            ,trainable=False))
# dropsout random neurons (ignores 20% of neurons)
model.add(Dropout(0.2))
# Convolutional Layer
model.add(Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(Conv1D(filters=128, kernel_size=4, padding='same', activation='relu'))
model.add(Conv1D(filters=128, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# %%
"""
### Compiling Model
"""

# %%
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
print(model.summary())


# %%
"""
## Section 4: Training the Model

The model is trained `EPOCHS` time, and Early Stopping argument is used to end training if the loss or accuracy don't improve within 3 epochs.
"""

# %%
early_stop2 = EarlyStopping(monitor='val_loss', patience=3)

data_train2 = data_train
labels_train2 = labels_train
data_val2 = data_val
labels_val2 = labels_val

hist2 = model.fit(data_train2, labels_train2,
                  validation_data=(data_val2, labels_val2),
                  epochs=EPOCHS, batch_size=32, shuffle=True,
                  callbacks=[early_stop2])

# %%
"""
### Results
"""

# %%
"""
Summarize history for accuracy
"""

# %%
plt.plot(hist2.history['acc'])
plt.plot(hist2.history['val_acc'])
plt.title('Multi Channel CNN model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# %%
"""
Summarize history for loss
"""

# %%
plt.plot(hist2.history['loss'])
plt.plot(hist2.history['val_loss'])
plt.title('Multi Channel CNN model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# %%
"""
Percentage accuracy of model
"""

# %%
data_test2 = data_test
labels_test2 = labels_test


labels_pred2 = model.predict(data_test2)
labels_pred2 = np.round(labels_pred2.flatten())
labels_pred2 = labels_pred2[0:2844]
accuracy2 = accuracy_score(labels_test2, labels_pred2)
print("Accuracy: %.2f%%" % (accuracy2*100))

# %%
"""
f1, precision, and recall scores
"""

# %%
print(classification_report(labels_test2, labels_pred2))

#######################################################
print("---------Multi Channel CNN ends here---------")
#######################################################

K.clear_session()
gc.collect()


#######################################################
print("---------Multi Channel Pooling CNN starts here---------")
#######################################################

# %%
model = Sequential()
# Embedded layer
model.add(Embedding(len(embedding_matrix), EMBEDDING_DIM,
                     weights=[embedding_matrix], 
                            input_length=MAX_SEQUENCE_LENGTH
                            ,trainable=False))
model.add(Dropout(0.2))
# Convolutional Layer
model.add(Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(Conv1D(filters=128, kernel_size=4, padding='same', activation='relu'))
model.add(Conv1D(filters=128, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(MaxPooling1D(pool_size=5))
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# %%
"""
### Compiling Model
"""

# %%
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
print(model.summary())


# %%
"""
## Section 4: Training the Model

The model is trained `EPOCHS` time, and Early Stopping argument is used to end training if the loss or accuracy don't improve within 3 epochs.
"""

# %%
early_stop3 = EarlyStopping(monitor='val_loss', patience=3)

data_train3 = data_train
labels_train3 = labels_train
data_val3 = data_val
labels_val3 = labels_val

hist3 = model.fit(data_train3, labels_train3, \
        validation_data=(data_val3, labels_val3), \
        epochs=EPOCHS, batch_size=32, shuffle=True, \
        callbacks=[early_stop3])

# %%
"""
### Results
"""

# %%
"""
Summarize history for accuracy
"""

# %%
plt.plot(hist3.history['acc'])
plt.plot(hist3.history['val_acc'])
plt.title('Multi Channel Pooling CNN model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# %%
"""
Summarize history for loss
"""

# %%
plt.plot(hist3.history['loss'])
plt.plot(hist3.history['val_loss'])
plt.title('Multi Channel Pooling CNN model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# %%
"""
Percentage accuracy of model
"""

# %%
data_test3 = data_test
labels_test3 = labels_test


labels_pred3 = model.predict(data_test3)
labels_pred3 = np.round(labels_pred3.flatten())
labels_pred3 = labels_pred3[0:2844]
accuracy3 = accuracy_score(labels_test3, labels_pred3)
print("Accuracy1: %.2f%%" % (accuracy3*100))

# %%
"""
f1, precision, and recall scores
"""

# %%
print(classification_report(labels_test3, labels_pred3))

#######################################################
print("---------Multi Channel Pooling CNN ends here---------")
#######################################################


K.clear_session()
gc.collect()


#######################################################
print("---------LSTM RNN starts here---------")
#######################################################
from keras.models import Sequential
from keras.layers import Dropout, Masking, Embedding, Bidirectional
from attention import Attention
# %%
model = Sequential()
# Embedded layer
model.add(Embedding(len(embedding_matrix), EMBEDDING_DIM,
                     weights=[embedding_matrix], 
                            input_length=MAX_SEQUENCE_LENGTH
                            ,trainable=False))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(100, return_sequences=True)))
model.add(Attention())
# LSTM Layer
#model3.add(LSTM(100))
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# %%
"""
### Compiling Model
"""

# %%
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
print(model.summary())


# %%
"""
## Section 4: Training the Model

The model is trained `EPOCHS` time, and Early Stopping argument is used to end training if the loss or accuracy don't improve within 3 epochs.
"""

# %%
early_stop4 = EarlyStopping(monitor='val_loss', patience=3)

data_train4 = data_train
labels_train4 = labels_train
data_val4 = data_val
labels_val4 = labels_val

hist4 = model.fit(data_train4, labels_train4, \
        validation_data=(data_val4, labels_val4), \
        epochs=EPOCHS, batch_size=32, shuffle=True, \
        callbacks=[early_stop4])

# %%
"""
### Results
"""

# %%
"""
Summarize history for accuracy
"""

# %%
plt.plot(hist4.history['acc'])
plt.plot(hist4.history['val_acc'])
plt.title('LSTM RNN model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# %%
"""
Summarize history for loss
"""

# %%
plt.plot(hist4.history['loss'])
plt.plot(hist4.history['val_loss'])
plt.title('LSTM RNN model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# %%
"""
Percentage accuracy of model
"""

# %%
data_test4 = data_test
labels_test4 = labels_test


labels_pred4 = model.predict(data_test4)
labels_pred4 = np.round(labels_pred4.flatten())
labels_pred4 = labels_pred4[0:2844]
accuracy4 = accuracy_score(labels_test4, labels_pred4)
print("Accuracy1: %.2f%%" % (accuracy4*100))

# %%
"""
f1, precision, and recall scores
"""

# %%
print(classification_report(labels_test4, labels_pred4))

#######################################################
print("---------LSTM RNN ends here---------")
#######################################################