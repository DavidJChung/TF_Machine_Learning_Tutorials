import nltk
from nltk.tokenize import word_tokenize         #tokenizes string into array
from nltk.stem import WordNetLemmatizer         #stemming removes postfixes such as ing, er
import numpy as np
import random
import pickle
from collections import Counter

lemmatizer = WordNetLemmatizer()
hm_lines = 10000000                           #memory errors can occur due to ram restrictions

def create_lexicon(pos,neg):
    lexicon = []                             #lexicon creation
    for fi in [pos, neg]:                       #calls files
        with open(fi, 'r') as f:                # openes files for reading
            contents = f.readlines()            #reads lines of open file
            for l in contents[:hm_lines]:       #for the limit of lines create tokens for each word and adds to lexicon
                all_words = word_tokenize(l.lower())
                lexicon += list(all_words)
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    w_counts = Counter(lexicon)

    l2 = []

    for w in w_counts:
            if 1000 > w_counts[w] > 50:         # removes common or rare words from lexicon
               l2.append(w)       # also allows for data set to be simple enough to process in as few layers as possible

    print(len(l2))
    return l2

def sample_handling(sample, lexicon, classification):
    featureset = []

    with open(sample,'r') as f:
        contents =f.readline()
        for l in contents[:hm_lines]:
            current_words = word_tokenize(l.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] += 1
            features = list(features)
            featureset.append([features, classification])
    return featureset

def create_feature_sets_and_labels(pos,neg,test_size = 0.1):
    lexicon = create_lexicon(pos, neg)
    features = []
    features += sample_handling('pos.txt',lexicon,[1,0])
    features += sample_handling('neg.txt', lexicon, [0, 1])
    random.shuffle(features)

    features = np.array(features)
    testing_size = int(test_size*len(features))

    train_x = list(features[:, 0][:-testing_size])
    train_y = list(features[:, 1][:-testing_size])

    test_x = list(features[:, 0][:-testing_size])
    test_y = list(features[:, 1][:-testing_size])

    return train_x,train_y, test_x, test_y


if __name__ == '__main__':
    train_x,train_y,test_x,test_y = create_feature_sets_and_labels('pos.txt', 'neg.txt')
    with open('sentiment_set.pickle','wb') as f:
        pickle.dump([train_x,train_y, test_x,test_y], f)