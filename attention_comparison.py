import numpy as np

def get_comparative_measures(X, attn_1, attn_2, n=20) :
    if type(X[0]) == list :
        X = [x for y in X for x in y]
    assert len(X) == len(attn_1), print(len(X), len(attn_1))
    assert len(X) == len(attn_2), print(len(X), len(attn_2))
    top_feat_1, top_feat_2 = np.argsort(attn_1)[-n:], np.argsort(attn_2)[-n:]
    top_feat_1 = set([X[i] for i in top_feat_1])
    top_feat_2 = set([X[i] for i in top_feat_2])
    return {
        "len" : len(X),
        "A|B" : len(top_feat_1 | top_feat_2),
        "A&B" : len(top_feat_1 & top_feat_2), 
        "A-B" : len(top_feat_1 - top_feat_2),
        "B-A" : len(top_feat_2 - top_feat_1),
        "jacc" : len(top_feat_1 & top_feat_2) / len(top_feat_1 | top_feat_2),
        'A' : sorted(list(top_feat_1)),
        'B' : sorted(list(top_feat_2)),
        'max_A' : max(attn_1),
        'max_B' : max(attn_2),
        'emd' : emd(attn_1, attn_2),
        'haus' : get_distance_between_top_10(attn_1, attn_2, n)
    }

def get_distance_between_top_10(a1, a2, n=20) :
    distance = 0
    top_10_1 = np.argsort(a1)[-n:]
    top_10_2 = np.argsort(a2)[-n:]
    for a in top_10_1:
        distance += min([abs(a-b) for b in top_10_2])

    return distance

def emd(a1, a2) :
    emds = []
    emd = 0
    for p, q in zip(a1, a2) :
        emd = p + emd - q
        emds.append(abs(emd))
    
    return emds, sum(emds)

def get_top_n_attended_words(X, attn, n=20) :
    if type(X[0]) == list :
        X = [x for y in X for x in y]
    assert len(X) == len(attn), print(len(X), len(attn))
    top_feat_1 = np.argsort(attn)[-n:]
    top_feat_1 = set([X[i] for i in top_feat_1])
    return top_feat_1

from collections import Counter
def get_common_attended_words(Xs, attns, n=20) :
    c = Counter()
    for X, attn in zip(Xs, attns) :
        f = list(get_top_n_attended_words(X, attn, n))
        c.update(f)
    return c