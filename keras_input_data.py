


def sent2ind(sent, word_idx_map):
    """
    Transforms sentence into a list of indices.
    """
    x = []
    words = str(sent).split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    return x
def make_idx_data(sentences, word_idx_map):
    """
    Transforms sentences (corpus, a list of sentence) into a 2-d matrix.
    """

    idx_data = []
    for sent in sentences:
        idx_sent = sent2ind(sent, word_idx_map)
        idx_data.append(idx_sent)
    # idx_data = np.array(idx_data, dtype=np.int)
    return idx_data






