from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
import numpy as np
from sklearn import cross_validation
from keras_input_data import make_idx_data
from load_vai import loadVAI
import _pickle as cPickle
from metrics import continuous_metrics
from keras.preprocessing.text import Tokenizer
import keras
import gensim
from sklearn.kernel_ridge import KernelRidge
from gensim.models.doc2vec import Doc2Vec , TaggedDocument
from sentences_getvec import getSentences_vec
from sklearn import datasets, linear_model
from sklearn.svm import SVR


if __name__ == '__main__':
    x = cPickle.load(open("mr.p", "rb"))
    revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]
    print("data loaded!")
    sentences=[]
    for rev in revs:
        sentence = rev['text']
        sentences.append(sentence)
    idx_data = make_idx_data(sentences, word_idx_map)
    #print(idx_data)

    dim = 'A'
    column = loadVAI(dim)
    irony=column
    maxlen = 87  # cut texts after this number of words (among top max_features most common words)
    batch_size = 8

    option = 'Irony'  # or Arousal,irony
    Y = np.array(irony)
    Y = [float(x) for x in Y]
    print(option + ' prediction.......................')

    # X_train, X_test, y_train, y_test = cross_validation.train_test_split(idx_data, Y, test_size=0.2,
    #                                                                      random_state=2)
    n_MAE=0
    n_Pearson_r=0
    n_Spearman_r=0
    n_MSE=0
    n_R2=0
    n_MSE_sqrt=0
    SEED = 42

    n = 5  # repeat the CV procedure 5 times to get more precise results
    for i in range(n):
        # for each iteration, randomly hold out 20% of the data as CV set
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(
            idx_data, Y, test_size=.20, random_state=i * SEED)

        print(len(X_train), 'train sequences')
        print(len(X_test), 'test sequences')
        print("Pad sequences (samples x time)")
        X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
        X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
        print('X_train shape:', X_train.shape)
        print('X_test shape:', X_test.shape)
        #print(X_train)
        X_train=np.array(getSentences_vec(X_train))
        print(X_train.shape)
        X_test =np.array(getSentences_vec(X_test))
        print(X_test.shape)
        """
        convert 3dim to 2dim
        """
        nsamples, nx, ny = X_train.shape
        X_train = X_train.reshape((nsamples, nx * ny))
        nsamples, nx, ny = X_test.shape
        X_test= X_test.reshape((nsamples, nx * ny))

        #print(X_test)

        print("")
        #print("Method = Linear ridge regression with doc2vec features")
        # model = KernelRidge(kernel='linear')
        # model.fit(X_train, y_train)
        #model = KernelRidge(kernel='linear')
        #model=linear_model.LinearRegression()
        model = SVR(C=1.0, epsilon=0.2)
        # print(X_train.shape)
        # print(len(y_train))
        model.fit(X_train, y_train)
        results = model.predict(X_test)
        # print('Y_test: %s' % str(y_test))
        # print('Predict value: %s' % str(predict))
        print(results)
        estimate=continuous_metrics(y_test, results, 'prediction result:')
        # MSE, MAE, Pearson_r, R2, Spearman_r, MSE_sqrt
    
        n_MAE += estimate[0]
        n_Pearson_r += estimate[1]
       
    ndigit=3
   
    avg_MAE =  round(n_MAE/5, ndigit)
    avg_Pearson_r =  round(n_Pearson_r/5, ndigit)
   
    print('average evaluate result:')
    print(avg_MAE ,avg_Pearson_r)