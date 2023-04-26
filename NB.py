import glob
import re
import nltk
import random
import string
import numpy as np
import collections
import itertools
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.metrics import precision, recall
from nltk.tokenize import TweetTokenizer
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

random.seed(33)
np.set_printoptions(precision=2)
stop_words = set(stopwords.words('english'))
port = PorterStemmer()
tknzr = TweetTokenizer()

# Emotion classes to be classified
classes = ['Angry', 'Happy', 'Relaxed', 'Sad']

# Dataset file lists, separated by emotion

happy_filelist = glob.glob('./data/Happy/Train/happy*.txt')
#happy_filelist += glob.glob('./data/Happy/Test/happy*.txt')
angry_filelist = glob.glob('./data/Angry/Train/angry*.txt')
#angry_filelist += glob.glob('./data/Angry/Test/angry*.txt')
relaxed_filelist = glob.glob('./data/Relaxed/Train/relaxed*.txt')
#relaxed_filelist += glob.glob('./data/Relaxed/Test/relaxed*.txt')
sad_filelist = glob.glob('./data/Sad/Train/sad*.txt')
#sad_filelist += glob.glob('./data/Sad/Test/sad*.txt')

# Reads all files from given filelist and associated emotion tag. 
# Returns two lists: list of tuples containing lyrics for each file and corresponding emotion
# as well as a list of all words contained in the files
def read(filelist, tag):
    lyrics = []
    all_words = []

    for f in filelist:
        try:
            with open(f,'r') as file:
                song = file.read()
                file.close()
                song = re.sub(r"(\\n|\\u....|\t)", "", song)
                song = re.sub(r"(\[\d\d:\d\d\.\d\d\])","",song)
                song = song.lower()
                song = nltk.word_tokenize(song)
                #song = tknzr.tokenize(song)
                song = [w for w in song if not w in string.punctuation]
                song = [w for w in song if not w in stop_words]
                song = [port.stem(w) for w in song]
                song_tag = (song, tag)
                lyrics.append(song_tag)

                for word in song:
                    all_words.append(word)
        except:
            break
    return lyrics, all_words

# Returns boolean dictionary of presence of word_features words in given song
def find_features(song):
    words = set(song)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

# Returns a plotted confusion matrix with color coding. 
# Normalized confusion matrix obtained by setting normalize=True
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    #else:
        #print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Read in songs for each of the emotions
happy_lyrics, happy_words = read(happy_filelist, 'happy')
angry_lyrics, angry_words = read(angry_filelist, 'angry')
relaxed_lyrics, relaxed_words = read(relaxed_filelist, 'relaxed')
sad_lyrics, sad_words = read(sad_filelist, 'sad')

# Store all words from read in files. Find frequency distribution of words and save top 100
all_words = happy_words + angry_words + relaxed_words + sad_words
all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:100]

# Shuffle each emotions lyrics
random.shuffle(happy_lyrics)
random.shuffle(angry_lyrics)
random.shuffle(relaxed_lyrics)
random.shuffle(sad_lyrics)

folds = 5
subset_size = 20
sum = 0
test_truth = []
test_predict = []

happy_prec = 0
angry_prec = 0
relaxed_prec = 0
sad_prec = 0
happy_rec = 0
angry_rec = 0
relaxed_rec = 0
sad_rec = 0

test_truth = []
test_predict = []

# Perform 5-Fold cross validation on songs with an 80/20 split for training and testing
# Classification is Naive Bayes
for i in range (folds):
    start = int(i*subset_size)
    end = int(start + subset_size)

    test_setHappy = happy_lyrics[start:end]
    train_setHappy = happy_lyrics[:start] + happy_lyrics[end:]

    test_setAngry = angry_lyrics[start:end]
    train_setAngry = angry_lyrics[:start] + angry_lyrics[end:]

    test_setRelaxed = relaxed_lyrics[start:end]
    train_setRelaxed = relaxed_lyrics[:start] + relaxed_lyrics[end:]

    test_setSad = sad_lyrics[start:end]
    train_setSad = sad_lyrics[:start] + sad_lyrics[end:]
    
    # combine training and testing data for each emotion
    training_docs = train_setHappy + train_setAngry + train_setRelaxed + train_setSad
    testing_docs = test_setHappy + test_setAngry + test_setRelaxed + test_setSad
    
    # find features for training and testing data
    train_set = [(find_features(song), tag) for (song, tag) in training_docs]
    test_set = [(find_features(song), tag) for (song, tag) in testing_docs]

    model = nltk.NaiveBayesClassifier.train(train_set)
    
    #mnb = SklearnClassifier(MultinomialNB())
    #model = mnb.train(train_set)
    #bnb = SklearnClassifier(BernoulliNB())
    #model = bnb.train(train_set)
    
    #lg = SklearnClassifier(LogisticRegression())
    #model = lg.train(train_set)
    
    #svc = SklearnClassifier(LinearSVC())
    #model = svc.train(train_set)
    
    #nsvc = SklearnClassifier(NuSVC())
    #model = nsvc.train(train_set)
    
    #rf = SklearnClassifier(RandomForestClassifier())
    #model = rf.train(train_set)
    
    # calculate accuracy for fold
    acc = nltk.classify.accuracy(model, test_set)
    sum += acc

    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)

    for j, (feats, label) in enumerate(test_set):
        refsets[label].add(j)
        observed = model.classify(feats)
        testsets[observed].add(j)
    
    # calculate and accumulate precision and recall for each emotion in fold
    happy_prec += precision(refsets['happy'], testsets['happy'])
    angry_prec += precision(refsets['angry'], testsets['angry'])
    relaxed_prec += precision(refsets['relaxed'], testsets['relaxed'])
    sad_prec += precision(refsets['sad'], testsets['sad'])
    happy_rec += recall(refsets['happy'], testsets['happy'])
    angry_rec += recall(refsets['angry'], testsets['angry'])
    relaxed_rec += recall(refsets['relaxed'], testsets['relaxed'])
    sad_rec += recall(refsets['sad'], testsets['sad'])

    # accumulate results for each fold - to build confuison matrix for all folds
    test_truth += [s  for (t,s) in test_set]
    test_predict += [model.classify(t) for (t,s) in test_set]

# Build confusion matrix
conf = confusion_matrix(test_truth, test_predict)
 
# Print ploted and color coded confusion matrix    
plot_confusion_matrix(conf, classes, normalize=True, title='Normalized Confusion Matrix')
plt.show()


# Print Averge Metrics for accuracy, precision and recall
print("Average Accuracy:", (sum/folds)*100)
print("Average Happy Precision:", (happy_prec/folds)*100)
print("Average Angry Precision:", (angry_prec/folds)*100)
print("Average Relaxed Precision:", (relaxed_prec/folds)*100)
print("Average Sad Precision:", (sad_prec/folds)*100)
print("Average Happy Recall:", (happy_rec/folds)*100)
print("Average Angry Recall:", (angry_rec/folds)*100)
print("Average Relaxed Recall:", (relaxed_rec/folds)*100)
print("Average Sad Recall:", (sad_rec/folds)*100)
