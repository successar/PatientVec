import torch
import torch.nn as nn

class SequenceClassifier(nn.Module) :
    def __init__(self, embedder, encoder, decoder, predictor) :
        super().__init__()
        self.embedding = embedder
        self.encoder = encoder
        self.decoder = decoder
        self.predictor = predictor

    def forward(self, data) :
        seq = data.seq
        lengths = data.lengths
        embedding = self.embedding(seq) #(B, L, E)
        h, hseq = self.encoder(embedding, lengths)
        h = data.correct(h)
        potential = self.decoder(h)

        weight = data.weight if hasattr(data, 'weight') else None
        target = data.target if hasattr(data, 'target') else None 

        data.predict, data.loss = self.predictor(potential, target, weight)