# DetectDepressionInTwitterPostsCNN-RNN


**Introduction:**

The detection of mental illness in social media, that reflects people’s life, is complex and it is continuously increasing in popularity on social media platforms. Supervised machine learning is not widely accepted because of the difficulties in obtaining sufficient amounts of annotated training data. The most effective deep neural networks architectures are identified and used to detect users with signs of mental illness given limited instructed text data.
Data:
We used two kinds of tweets that are needed for this project: random tweets that do not indicate depression and tweets that show the user may have depression. The random tweets dataset can be found from the Kaggle dataset twitter_sentiment. It is harder to get tweets that indicate depression as there is no public dataset of depressive tweets, so in this project tweets indicating depression are retrieved using the Twitter scraping tool TWINT using the keyword depression by scraping all tweets in an one day span. The scrapped tweets may contain tweets that do not indicate the user having depression, such as tweets linking to articles about depression. In addition, the pretrained vectors for the Word2Vec model is from here. The data consists of 3842 tweets labelled as Depressed and 1048588 random tweets


**Our objective:**

Is to detect depression using the most effective deep neural architecture from two of the most popular deep learning approaches in the field of natural language processing: Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs), given the limited amount of unstructured data.
Our approach and key contributions can be summarized as follows.
• We focus on identifying users suffering from depression based on their social posts such as tweets.


**Job Flow:**

First, we start by importing the data and the pretrained Word2Vec Model.
After that we removed all the retweets, URL’s, @mentions, and all the non-alphanumeric characters. Also, all the stop words except for first, second, and third person pronouns were removed, And get rid of stuff like “what's” and making it “what is”.
Then we used a Tokenizer to assign indices and filtering out unfrequent words. We only care about the top 20000 most frequent words.
Then we Apply the tokenizer to depressive tweets and random tweets data.
After that, padding sequences all to the same length of 140 words.
The embedding matrix is a `n x m` matrix where `n` is the number of words and `m` is the dimension of the embedding. In this case, `m=300` and `n=20000`. We take the min between the number of unique words in our tokenizer and max words in case there are less unique words than the max we specified.
After that we split and format the data, assigning labels to the depressive tweets and random tweets data, and splitting the arrays into test (60%), validation (20%), and train data (20%). Combine depressive tweets and random tweets arrays and shuffle.
We used four neural network models to evaluate the performance, three of them use CNN and one uses RNN. A drop out of a probability 0.2 follows the word embedding layer. Each model is followed by a vanilla layer that is fully-connected, has 250 hidden units, and uses a Rectified Linear Unit (ReLU) activation. Then, we apply dropout with a probability of 0.2. The output layer is a fully-connected layer with one hidden unit, and it uses a sigmoid activation to produce an output.

1)	CNN With Max – applying a one-dimensional convolution operation with 250 filters and a kernel of size 3. After that max pooling is applied to extract global          abstract which results in an abstract feature representation of length 250.
2)	Multi Channel CNN - We apply 3 convolutions, each of which has 128 features and filters of the lengths 3, 4, and 5. A one-dimensional operation is used. Then, a max-pooling layer is applied on the feature map to extract abstract information. Finally, we concatenate feature representations into a single output. Conversely to recurrent layers, convolutional operations are helpful with max-pooling to extract word features without considering the sequence order
3)	Multi Channel Pooling CNN – extension of the Multi Channel CNN with applying two different max pooling sizes, 2 and 5.
4)	Bidirectional LSTM with attention – we use bidirectional LSTM layers with 100 units.


**Obtained results:**

| Model | Accuracy || Precision | Recall | F1 |
| --- | --- |--- | --- |--- | --- |
| CNNWithMax | 98.77% | 0 - 1 | 0.99 - 0.98 | 1.00 - 0.94 | 0.99 - 0.96 |
| MultiChannelCNN | 83.23% | 0 - 1 | 0.84 - 0.27 | 0.99 - 0.02 | 0.91 - 0.04 |
| MultiChannelPoolingCNN | 81.43% | 0 - 1 | 0.84 - 0.16 | 0.97 - 0.03 | 0.90 - 0.05 |
| BiLSTM | 98.59% | 0 - 1 | 0.99 - 0.98 | 0.99 - 0.93 | 0.99 - 0.96 |



**Conclusion:**
In conclusion, we performed a comparative evaluation on 4 different deep learning models for depression detection from tweets on the user level.
We performed the experiments on Kaggle’s dataset twitter_sentiment and using the Twitter scraping tool TWINT using the keyword depression by scraping all tweets in a one day span.
The experiments showed that using CNN with Max model got us the best results followed by RNN with Bidirectional LSTM model.



**Required Libraries**

    ftfy - fixes Unicode that's broken in various ways
    gensim - enables storing and querying word vectors
    keras - a high-level neural networks API running on top of TensorFlow
    matplotlib - a Python 2D plotting library which produces publication quality figures
    nltk - Natural Language Toolkit
    numpy - the fundamental package for scientific computing with Python
    pandas - provides easy-to-use data structures and data analysis tools for Python
    sklearn - a software machine learning library
    tensorflow - an open source machine learning framework for everyone
