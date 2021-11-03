import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LSTMSarcasm(nn.module):
    def __init__(self, embed_dim, hidden_dim, vocab_size):
        #initialize some variables
        self.hidden_dim=hidden_dim
        self.embed_dim=embed_dim
        self.vocab_size=vocab_size

        #form the layers of the network
        self.word_embed = nn.embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim)
        #predictions are either sarcasm or not, so two labels
        self.out = nn.Linear(hidden_dim, 2)

    def forward(self,tweet):
        #embed the tweet
        embed = self.word_embed(tweet)
        #lstm layer
        lstm_out, _ = self.lstm(embed.view(len(tweet),1,-1))
        #output connected layer
        outlayer = self.out(lstm_out.view(len(tweet),-1))
        #softmax
        out = F.softmax(outlayer,dim=-1)


