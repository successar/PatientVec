import numpy as np

def get_comparative_measures(X, attn_1, attn_2, n=20) :
    top_feat_1, top_feat_2 = np.argsort(attn_1)[-n:], np.argsort(attn_2)[-n:]
    top_feat_1 = set([X[i] for i in top_feat_1])
    top_feat_2 = set([X[i] for i in top_feat_2])
    return {
        "A&B" : len(top_feat_1 & top_feat_2), 
        "A-B" : len(top_feat_1 - top_feat_2),
        "B-A" : len(top_feat_2 - top_feat_1)
    }