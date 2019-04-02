import torch
import torch.nn as nn

from .modules.Encoder import Encoder
from .modules.Embedder import Embedder
from .modules.Feedforward import FeedForward
from .modules.Predictor import Predictor

from allennlp.common import from_params
from .Model import Model

from .modules.SelfAttention import SelfAttention
from .modules.Attention import Attention

from .utils import Holder, HierHolder
from typing import Dict
from allennlp.common import Params

@Model.register("vec_classifier")
class VectorClassifier(nn.Module, from_params.FromParams) :
    def __init__(self, decoder: FeedForward, 
                        predictor: Predictor, 
                        structured: Dict,
                        reg: Dict) :
        super().__init__()
        self.decoder = decoder
        self.predictor = predictor
        self.structured = structured
        self.reg = reg

    def forward(self, batch) :
        h = torch.Tensor(batch.X).cuda()

        if self.structured['use_structured'] :
            conditional = torch.Tensor(batch.structured_data).cuda()
            h = torch.cat([h, conditional], dim=-1)

        potential = self.decoder(h)

        target = torch.Tensor(batch.y).cuda() if batch.have('y') else None
        weight = batch.weight if batch.have('weight') else None

        predict, loss = self.predictor(potential, target, weight)

        if self.reg['type'] == 'l1' :
            loss += self.reg['weight'] * torch.abs(self.decoder._linear_layers[-1].weight).sum()

        batch.outputs = { "predict" : predict, "loss" : loss }

    @classmethod
    def from_params(cls, params: Params) :
        if params['structured']['use_structured'] :
            params['decoder']['input_dim'] += params['structured']['structured_dim']

        decoder = FeedForward.from_params(params.pop('decoder'))
        predictor = Predictor.from_params(params.pop('predictor'))

        return cls(decoder=decoder, predictor=predictor, structured=params.pop('structured'), reg=params.pop('reg'))

@Model.register("seq_classifier")
class SequenceClassifier(nn.Module, from_params.FromParams) :
    def __init__(self, embedder: Embedder, 
                        encoder: Encoder, 
                        decoder: FeedForward, 
                        predictor: Predictor, 
                        structured: Dict) :
        super().__init__()
        self.embedding = embedder
        self.encoder = encoder
        self.decoder = decoder
        self.predictor = predictor
        self.structured = structured

    def forward(self, batch) :
        data = Holder(batch.X)
        seq = data.seq
        lengths = data.lengths
        
        embedding = self.embedding(seq) #(B, L, E)
        h, hseq = self.encoder(embedding, lengths)

        if self.structured['use_structured'] :
            conditional = torch.Tensor(batch.structured_data).cuda()
            h = torch.cat([h, conditional], dim=-1)

        potential = self.decoder(h)

        target = torch.Tensor(batch.y).cuda() if batch.have('y') else None
        weight = batch.weight if batch.have('weight') else None

        if self.predictor.replicate :
            if self.structured['use_structured'] :
                conditional = torch.Tensor(batch.structured_data).cuda()
                hseq = torch.cat([hseq, conditional.unsqueeze(1).expand(-1, hseq.shape[1], -1)], dim=-1)
            potential_seq = self.decoder(hseq)
            predict, loss = self.predictor(potential, target, weight, masks=data.masks, potential_seq=potential_seq)
        else :
            predict, loss = self.predictor(potential, target, weight)
        batch.outputs = { "predict" : predict, "loss" : loss }

    @classmethod
    def from_params(cls, params: Params) :
        embedder = Embedder.from_params(params.pop('embedder'))
        encoder = Encoder.from_params(input_size=embedder.embed_size, params=params.pop('encoder'))

        params['decoder']['input_dim'] = encoder.output_size
        if params['structured']['use_structured'] :
            params['decoder']['input_dim'] += params['structured']['structured_dim']

        decoder = FeedForward.from_params(params.pop('decoder'))
        predictor = Predictor.from_params(params.pop('predictor'))

        return cls(embedder=embedder, encoder=encoder, decoder=decoder, predictor=predictor, structured=params.pop('structured'))
        
@Model.register("seq_classifier_with_attention")
class SequenceClassifierWithAttention(nn.Module, from_params.FromParams) :
    def __init__(self, embedder: Embedder, 
                       encoder: Encoder, 
                       decoder: FeedForward, 
                       attention: SelfAttention, 
                       predictor: Predictor, 
                       structured: Dict) :
        super().__init__()
        self.embedding = embedder
        self.encoder = encoder
        self.decoder = decoder
        self.predictor = predictor
        self.attention = attention
        self.structured = structured

    def forward(self, batch) :
        data = Holder(batch.X)
        seq = data.seq
        lengths = data.lengths
        
        embedding = self.embedding(seq) #(B, L, E)
        h, hseq = self.encoder(embedding, lengths) #(B, H), #(B, L, H)
        
        attn = self.attention(hseq, data.masks) #(B, L)
        mix = torch.bmm(attn.unsqueeze(1), hseq).squeeze(1)

        if self.structured['use_structured'] :
            conditional = torch.Tensor(batch.structured_data).cuda()
            mix = torch.cat([mix, conditional], dim=-1)

        potential = self.decoder(mix)

        target = torch.Tensor(batch.y).cuda() if batch.have('y') else None
        weight = batch.weight if batch.have('weight') else None

        predict, loss = self.predictor(potential, target, weight)
        batch.outputs = { "predict" : predict, "loss" : loss, "attention" :  data.depad(attn) }

    @classmethod
    def from_params(cls, params: Params) :
        embedder = Embedder.from_params(params.pop('embedder'))
        encoder = Encoder.from_params(input_size=embedder.embed_size, params=params.pop('encoder'))

        params['attention']['similarity']['tensor_1_dim'] = encoder.output_size
        attention = SelfAttention.from_params(params.pop('attention'))

        params['decoder']['input_dim'] = encoder.output_size
        if params['structured']['use_structured'] :
            params['decoder']['input_dim'] += params['structured']['structured_dim']

        decoder = FeedForward.from_params(params=params.pop('decoder'))
        predictor = Predictor.from_params(params.pop('predictor'))

        return cls(embedder=embedder, encoder=encoder, decoder=decoder, attention=attention, predictor=predictor, structured=params.pop('structured'))

@Model.register("seq_classifier_with_structured_attention")
class SequenceClassifierWithStructuredAttention(nn.Module, from_params.FromParams) :
    def __init__(self, embedder: Embedder, 
                       encoder: Encoder, 
                       decoder: FeedForward, 
                       attention: Attention, 
                       predictor: Predictor, 
                       structured: Dict) :
        super().__init__()
        self.embedding = embedder
        self.encoder = encoder
        self.decoder = decoder
        self.predictor = predictor
        self.attention = attention
        self.structured = structured

    def forward(self, batch) :
        data = Holder(batch.X)
        seq = data.seq
        lengths = data.lengths
        
        embedding = self.embedding(seq) #(B, L, E)
        h, hseq = self.encoder(embedding, lengths) #(B, H), #(B, L, H)
        
        conditional = torch.Tensor(batch.cond).cuda().unsqueeze(1)
        attn = self.attention(hseq, conditional, data.masks) #(B, L)
        mix = torch.bmm(attn.unsqueeze(1), hseq).squeeze(1)
        
        if self.structured['use_structured'] :
            conditional = torch.Tensor(batch.structured_data).cuda()
            mix = torch.cat([mix, conditional], dim=-1)

        potential = self.decoder(mix)

        target = torch.Tensor(batch.y).cuda() if batch.have('y') else None
        weight = batch.weight if batch.have('weight') else None

        predict, loss = self.predictor(potential, target, weight)
        batch.outputs = { "predict" : predict, "loss" : loss, "attention" :  data.depad(attn) }

    @classmethod
    def from_params(cls, params: Params) :
        embedder = Embedder.from_params(params.pop('embedder'))
        encoder = Encoder.from_params(input_size=embedder.embed_size, params=params.pop('encoder'))

        params['attention']['similarity']['tensor_1_dim'] = encoder.output_size
        attention = Attention.from_params(params.pop('attention'))

        params['decoder']['input_dim'] = encoder.output_size
        if params['structured']['use_structured'] :
            params['decoder']['input_dim'] += params['structured']['structured_dim']

        decoder = FeedForward.from_params(params=params.pop('decoder'))
        predictor = Predictor.from_params(params.pop('predictor'))

        return cls(embedder=embedder, encoder=encoder, decoder=decoder, attention=attention, predictor=predictor, structured=params.pop('structured'))

@Model.register("hierarchical_classifier_with_attention")
class HierarchicalClassifierWithAttention(nn.Module, from_params.FromParams) :
    def __init__(self, embedder: Embedder, 
                       word_encoder: Encoder, 
                       word_attention: SelfAttention, 
                       sentence_encoder: Encoder, 
                       sentence_attention: SelfAttention,
                       decoder: FeedForward, 
                       predictor: Predictor, 
                       structured: Dict) :
        super().__init__()
        self.embedding = embedder
        self.word_encoder = word_encoder
        self.sentence_encoder = sentence_encoder

        self.word_attention = word_attention
        self.sentence_attention = sentence_attention

        self.decoder = decoder
        self.predictor = predictor

        self.structured = structured

    def forward(self, batch) :
        data = HierHolder(batch.X)
        flatten_data = data.flatten_holder
        
        embedding = self.embedding(flatten_data.seq) #(B, L, E)
        h_w, hseq_w = self.word_encoder(embedding, flatten_data.lengths) #(B, H), #(B, L, H)
        
        attn_w = self.word_attention(hseq_w, flatten_data.masks) #(B, L)
        mix_w = torch.bmm(attn_w.unsqueeze(1), hseq_w).squeeze(1) #(B*S, H)
        unflatten_mix = data.unflatten(mix_w) #(B, S, H)

        h_s, hseq_s = self.sentence_encoder(unflatten_mix, data.doclens)

        attn_s = self.sentence_attention(hseq_s, data.flatten_mask)
        mix_s = torch.bmm(attn_s.unsqueeze(1), hseq_s).squeeze(1)

        if self.structured['use_structured'] :
            conditional = torch.Tensor(batch.structured_data).cuda()
            mix_s = torch.cat([mix_s, conditional], dim=-1)

        potential = self.decoder(mix_s)

        target = torch.Tensor(batch.y).cuda() if batch.have('y') else None
        weight = batch.weight if batch.have('weight') else None

        predict, loss = self.predictor(potential, target, weight)

        word_attentions = flatten_data.depad(attn_w)
        batch.outputs = { 
            "predict" : predict, 
            "loss" : loss, 
            "word_attention" : data.unflatten_list(word_attentions), 
            "sentence_attention" : data.depad(attn_s)
        }

    @classmethod
    def from_params(cls, params:Params) :
        embedder = Embedder.from_params(params.pop('embedder'))
        word_encoder = Encoder.from_params(input_size=embedder.embed_size, params=params.pop('word_encoder'))

        params['word_attention']['similarity']['tensor_1_dim'] = word_encoder.output_size
        word_attention = SelfAttention.from_params(params.pop('word_attention'))

        sentence_encoder = Encoder.from_params(input_size=word_encoder.output_size, params=params.pop('sentence_encoder'))

        params['sentence_attention']['similarity']['tensor_1_dim'] = sentence_encoder.output_size
        sentence_attention = SelfAttention.from_params(params.pop('sentence_attention'))

        params['decoder']['input_dim'] = sentence_encoder.output_size
        if params['structured']['use_structured'] :
            params['decoder']['input_dim'] += params['structured']['structured_dim']

        decoder = FeedForward.from_params(params=params.pop('decoder'))
        predictor = Predictor.from_params(params.pop('predictor'))

        return cls(embedder=embedder, 
                    word_encoder=word_encoder, 
                    word_attention=word_attention, 
                    sentence_encoder=sentence_encoder, 
                    sentence_attention=sentence_attention,
                    decoder=decoder, 
                    predictor=predictor, 
                    structured=params.pop('structured'))