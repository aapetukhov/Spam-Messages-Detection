import torch
import torch.nn as nn
from torch.nn.functional import softmax
import math


class PositionalEncoder(nn.Module):
    # from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, max_length: int = 171, embed_dim: int = 100, dropout: float = 0.2):
        super().__init__()
        self.pos_features = torch.zeros(max_length, embed_dim)

        positions = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        freqs = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float) * \
                          (-math.log(10000) / embed_dim)).unsqueeze(0)
        print(freqs)

        arguments = positions * freqs
        self.pos_features[:, 0::2] = torch.sin(arguments)
        self.pos_features[:, 1::2] = torch.cos(arguments)
        self.pos_features = self.pos_features.unsqueeze(0)
        self.pos_features = nn.Parameter(self.pos_features, requires_grad=False)
        # pos_features: (1, max_length, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        outputs = inputs + self.pos_features[:, :inputs.shape[1]]
        return self.dropout(outputs)


class Attention(nn.Module):
    # One head of attention
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        attention_dim = embed_dim // num_heads

        self.WQ = nn.Linear(embed_dim, attention_dim, bias=False)
        self.WK = nn.Linear(embed_dim, attention_dim, bias=False)
        self.WV = nn.Linear(embed_dim, attention_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.WQ.weight)
        nn.init.xavier_uniform_(self.WK.weight)
        nn.init.xavier_uniform_(self.WV.weight)

    def forward(self, query, key, value, mask=None):
        # query, key, value: (batch_size, length, embed_dim)
        # mask: (batch_size, length, length)

        Q = self.WQ(query)
        K = self.WK(key)
        V = self.WV(value)
        # Q, K, V: (batch_size, length, attention_dim)

        norm_factor = math.sqrt(Q.shape[-1])
        dot_products = torch.bmm(Q, K.transpose(1, 2)) / norm_factor
        # dot_products: (batch_size, length, length)

        if mask is not None:
            dot_products = dot_products.masked_fill(mask, -math.inf)

        attention_score = softmax(dot_products, dim=-1) #attention matrix. shows "importances" of words for each other in the sentence
        attention = torch.bmm(self.dropout(attention_score), V) #final attention layer
        # attention_score: (batch_size, length, length)
        # attention: (batch_size, length, attention_dim)

        return attention, attention_score


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super(MultiHeadAttention, self).__init__()

        assert embed_dim % num_heads == 0
        self.attention_heads = nn.ModuleList([Attention(embed_dim, num_heads, dropout)
                                              for _ in range(num_heads)])

        self.linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, query, key, value, mask=None):
        # shapes:
        # query, key, value: (batch_size, length, embed_dim)
        # mask: (batch_size, length, length)
        attentions, attention_scores = [], []

        for head in self.attention_heads:
            attention, attention_score = head(query, key, value, mask)
            attentions += [attention]
            attention_scores += [attention_score]

        attentions = torch.cat(attentions, dim=-1) #concatenating attention heads
        attention_scores = torch.stack(attention_scores, dim=-1)
        # attentions: (batch_size, length, embed_dim)
        # attention_scores: (batch_size, length, length, num_heads)

        outputs = self.linear(attentions)
        outputs = self.dropout(outputs)
        # outputs: (batch_size, length, embed_dim)

        return outputs, attention_scores
    

class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, fc_dim, num_heads, dropout):
        super().__init__()
        self.self_attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.feedforward[0].weight)
        nn.init.xavier_normal_(self.feedforward[3].weight)

    def forward(self, inputs, mask):
        attention, attention_score = self.self_attention(query=inputs, key=inputs, 
                                                         value=inputs, mask=mask)
        outputs = inputs + attention #skip-connection
        outputs = self.norm1(outputs)
        outputs = outputs + self.feedforward(outputs) #skip-connection
        outputs = self.norm2(outputs)
        return outputs, attention_score


class Transformer(nn.Module):
    def __init__(self, max_length, num_classes,
                 embed_dim, fc_dim, num_heads, num_layers, dropout, embedding_tensor):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding.from_pretrained(embedding_tensor, freeze=True, padding_idx=0) #using pretrained embeddings, not making them trainable
        self.pos_encoder = PositionalEncoder(max_length, embed_dim, dropout)
        self.layers = nn.ModuleList([TransformerLayer(embed_dim, fc_dim, num_heads, dropout) \
                                     for _ in range(num_layers)])
        self.classifier = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.classifier.weight)

    def forward(self, tokens):
        # source: (batch_size, length)
        embeds = self.embedding(tokens) * math.sqrt(self.embed_dim)
        outputs = self.pos_encoder(embeds)
        # outputs: (batch_size, length, embed_dim)
        check_for_nan_inf(outputs, 'Positional Encodings') #checking if eveything's alright

        padding_mask = create_padding_mask(tokens)
        attention_scores = []
        for layer in self.layers:
            outputs, attention_score = layer(outputs, padding_mask)
            attention_scores += [attention_score]
        attention_scores = torch.stack(attention_scores, dim=-1)
        # outputs: (batch_size, length, embed_dim)
        # attentions_scores: (batch_size, length, embed_dim, num_heads, num_layers)

        mask = (tokens != 0).to(torch.float).detach() #0 = padding index
        lengths = mask.sum(dim=1).detach()
        if (lengths == 0).any():
            print("Zero lengths detected, handling zero lengths.")
            lengths[lengths == 0] = 1.0 

        outputs = (outputs * mask.unsqueeze(2)).sum(dim=1) / (lengths.unsqueeze(1) + 1e-8)
        # outputs: (batch_size, embed_dim)
        check_for_nan_inf(outputs, 'Pooled Outputs')

        logits = self.classifier(outputs)
        # logits: (batch_size, num_classes)
        check_for_nan_inf(logits, 'Logits')

        return logits, attention_scores
    

def create_padding_mask(tokens, pad_idx=0):
    # tokens: (batch_size, length)
    length = tokens.shape[-1]
    padding_mask = (tokens == pad_idx)
    padding_mask = padding_mask.unsqueeze(1).repeat(1, length, 1)
    # padding_mask: (batch_size, length, length)

    return padding_mask


def check_for_nan_inf(tensor, name="Tensor"):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
    if torch.isinf(tensor).any():
        print(f"Inf detected in {name}")