from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.constraints import unitnorm
from keras.layers.core import Reshape, Flatten, Merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Convolution1D, MaxPooling1D
from sklearn.cross_validation import KFold
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
import numpy as np
from sklearn import cross_validation
import math
from keras_input_data import make_idx_data
from load_vai import loadVAI
import _pickle as cPickle
from metrics import continuous_metrics


def cnn(W=None):
    # Number of feature maps (outputs of convolutional layer)
    N_fm = 100
    dense_nb = 20
    # kernel size of convolutional layer
    kernel_size = 5
    conv_input_height = maxlen  # maxlen of sentence is 87
    conv_input_width = W.shape[1]  # dims=300
    global maxlen  #maxlen of sentence is 87


    model = Sequential()
    # Embedding layer (lookup table of trainable word vectors)
    model.add(Embedding(input_dim=W.shape[0], output_dim=W.shape[1], weights=[W], W_constraint=unitnorm()))
    # Reshape word vectors from Embedding to tensor format suitable for Convolutional layer
    #print(conv_input_height)
    #print(conv_input_width)
    #model.add(Reshape(dims=(1, conv_input_height, conv_input_width)))
    #model.add(Reshape((conv_input_height, conv_input_width)))
    model.add(Reshape(W.shape[0],(conv_input_height, conv_input_width)))
    #model.add(Reshape((test_size, 1, img_h, Words.shape[1])))

    # first convolutional layer
    model.add(Convolution2D(nb_filter=N_fm, nb_row=kernel_size, nb_col=conv_input_width, border_mode='valid',
                            W_regularizer=l2(0.0001), activation='relu'))
    # ReLU activation
    model.add(Dropout(0.5))

    # aggregate data in every feature map to scalar using MAX operation
    # model.add(MaxPooling2D(pool_size=(conv_input_height-kernel_size+1, 1), border_mode='valid'))
    model.add(MaxPooling2D(pool_size=(conv_input_height - kernel_size + 1, 1), border_mode='valid'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(output_dim=dense_nb, activation='relu'))
    model.add(Dropout(0.5))
    # Inner Product layer (as in regular neural network, but without non-linear activation function)
    model.add(Dense(output_dim=1, activation='linear'))
    # SoftMax activation; actually, Dense+SoftMax works as Multinomial Logistic Regression
    return model


def lstm(W):
    model = Sequential()
    model.add(Embedding(W.shape[0], W.shape[1], input_length=maxlen))
    model.add(LSTM(128))  # try using a GRU instead, for fun
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('linear'))

    return model

def imdb_cnn(W=None):
    # Number of feature maps (outputs of convolutional layer)
    N_fm = 100
    # kernel size of convolutional layer
    kernel_size = 5
    dims = 300  # 300 dimension
    maxlen = 87  # maxlen of sentence
    max_features = W.shape[0]
    hidden_dims = 100
    print('Build model...')
    model = Sequential()

    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    model.add(Embedding(max_features, dims, input_length=maxlen, weights=[W]))
    model.add(Dropout(0.5))

    # we add a Convolution1D, which will learn nb_filter
    # word group filters of size filter_length:
    model.add(Convolution1D(nb_filter=N_fm,
                            filter_length=kernel_size,
                            border_mode='valid',
                            activation='relu',
                            ))
    model.add(Dropout(0.4))
    # we use standard max pooling (halving the output of the previous layer):
    #model.add(MaxPooling1D(pool_length=math.floor((maxlen - kernel_size + 1) / 2)))
    model.add(MaxPooling1D(pool_length=2))

    # We flatten the output of the conv layer,
    # so that we can add a vanilla dense layer:
    model.add(Flatten())

    # We add a vanilla hidden layer:
    model.add(Dense(hidden_dims))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))

    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(1))
    model.add(Activation('linear'))
    return model
def cnn_lstm(W):
    nb_filter = 100
    filter_length = 5
    pool_length = 2
    lstm_output_size = 100
    p = 0.25

    model = Sequential()
    model.add(Embedding(W.shape[0], W.shape[1], input_length=maxlen, weights=[W]))
    model.add(Dropout(p))
    model.add(Convolution1D(nb_filter=nb_filter,
                            filter_length=filter_length,
                            border_mode='valid',
                            activation='relu',
                            subsample_length=1))
    model.add(MaxPooling1D(pool_length=pool_length))
    model.add(LSTM(lstm_output_size))
    model.add(Dense(lstm_output_size))
    model.add(Dropout(p))
    model.add(Dense(1))
    model.add(Activation('linear'))

    return model

# def lstm_cnn(W):
    # region_input = Input(shape=(maxlen,), dtype='int32', name='region_input')
    #     ###这是一个逗号标志的输入的区域的句子，属于整个文章的一个区域。
    # x = Embedding(W.shape[0], W.shape[1], weights=[W], input_length=maxlen)(region_input)

    # lstm_output = LSTM(64, return_sequences=True, name='lstm'), merge_mode='concat'(x)  

    # region_conv = Convolution1D(nb_filter=nb_filter,
    #                                 filter_length=filter_length,
    #                                 border_mode='valid',
    #                                 activation='relu',
    #                                 subsample_length=1)(lstm_output)
    # region_max = MaxPooling1D(pool_length=maxlen - filter_length + 1)(region_conv)
    # region_vector = Flatten()(region_max)
    # textvector = Dense(64, activation='relu')(region_vector)
    # predictions = Dense(1, activation='sigmoid')(textvector)
    # final_model = Model(region_input, predictions, name='model')

    # return model


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

    dim = 'I'
    column = loadVAI(dim)
    irony=column
    maxlen = 87  # cut texts after this number of words (among top max_features most common words)
    batch_size = 8

    # option = 'Irony'  # or Arousal,irony
    Y = np.array(irony)
    Y = [float(x) for x in Y]
    # print(option + ' prediction.......................')

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

        max_features = W.shape[0]  # shape of W: (13631, 300) , changed to 14027 through min_df = 3
        # print(max_features)

        print("Pad sequences (samples x time)")
        X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
        X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
        print('X_train shape:', X_train.shape)
        print('X_test shape:', X_test.shape)





        model = lstm_cnn(W)
        print('-----------lstm_cnn----------')
        model.compile(loss='mse', optimizer='adagrad')  # loss function: mse
        print("Train...")
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        result = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=10,validation_data=(X_test, y_test),
                           callbacks=[early_stopping])
        score = model.evaluate(X_test, y_test, batch_size=batch_size)
        print('Test score:', score)
        # experiment evaluated by multiple metrics
        predict = model.predict(X_test, batch_size=batch_size).reshape((1, len(X_test)))[0]
        # print('Y_test: %s' % str(y_test))
        # print('Predict value: %s' % str(predict))
        estimate=continuous_metrics(y_test, predict, 'prediction result:')
        # MSE, MAE, Pearson_r, R2, Spearman_r, MSE_sqrt
      
        n_MAE += estimate[1]
        n_Pearson_r += estimate[2]
      
    ndigit=3

    avg_MAE =  round(n_MAE/5, ndigit)
    avg_Pearson_r =  round(n_Pearson_r/5, ndigit)
 
    print('average evaluate result:')
    print(avg_MAE ,avg_Pearson_r)



    # predict = np.array([5] * len(y_test))
    # from metrics import continuous_metrics
    # continuous_metrics(y_test, predict, 'prediction result:')

    # visualization
    #from visualize import draw_linear_regression

    # X = range(50, 100)  # or range(len(y_test))
    # draw_linear_regression(X, np.array(y_test)[X], np.array(predict)[X], 'Sentence Number', option,
                              # 'Comparison of predicted and true ' + option)

    from visualize import plot_keras, draw_hist

    # plot_keras(result, x_labels='Epoch', y_labels='Loss')
    #draw_hist(np.array(y_test) - np.array(predict), title='Histogram of ' + option + ' prediction: ')
