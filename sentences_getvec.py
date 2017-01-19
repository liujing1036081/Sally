import _pickle as cPickle


x = cPickle.load(open("mr.p", "rb"))
revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]

def getSentences_vec(idx_data):
    sentences_vec = []

    for sentence in idx_data:
        x = []
        for index in sentence:
            word_vec=W[index]
            x.append(word_vec)
        sentence_vec=x
        sentences_vec.append(sentence_vec)
    # print(len(sentence_vec))
    # print(len(sentences_vec))
    return sentences_vec
