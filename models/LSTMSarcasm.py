import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LSTMSarcasm(nn.Module):
    def __init__(self, embed_dim, hidden_dim, vocab_size):
        super(LSTMSarcasm, self).__init__()
        #initialize some variables
        self.hidden_dim=hidden_dim
        self.embed_dim=embed_dim
        self.vocab_size=vocab_size

        #form the layers of the network
        self.word_embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, batch_first=True)
        #predictions are either sarcasm or not, so two labels
        self.outlayer = nn.Linear(hidden_dim, 1)
        self.sig = nn.Sigmoid()


    def forward(self,tweet):

        batch_size = tweet.size(0)

        #embed the tweet
        embed = self.word_embed(tweet)
        #lstm layer
        # lstm_out, _ = self.lstm(embed)
        #
        # lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        #
        # #output connected layer
        # outlayer = self.outlayer(lstm_out)
        # #softmax
        # out = self.sig(outlayer)
        #
        # out = out.view(batch_size, -1)
        # out = out[:,-1]

        lstm_out, (final,final_cell) = self.lstm(embed)

        # lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        #output connected layer
        outlayer = self.outlayer(final)
        #softmax
        out = self.sig(outlayer)

        out = out.view(batch_size, -1)
        out = out[:,-1]
        return out


