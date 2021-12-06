import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class LSTM(nn.Module):
    
    def __init__(self, embed_dim, hidden_dim, vocab_size):
        super(LSTM, self).__init__()
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


class LSTMAtt(nn.Module):
    
    def __init__(self, embed_dim, hidden_dim, vocab_size,seq_len):
        super(LSTMAtt, self).__init__()
        #initialize some variables
        self.hidden_dim=hidden_dim
        self.embed_dim=embed_dim
        self.vocab_size=vocab_size
        self.seq_len=seq_len

        #form the layers of the network
        self.word_embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, batch_first=True)
        #predictions are either sarcasm or not, so two labels
        self.outlayer = nn.Linear(hidden_dim, 1)
        self.sig = nn.Sigmoid()

        self.w_att = nn.Parameter(torch.zeros(hidden_dim,hidden_dim))
        self.b_att = nn.Parameter(torch.zeros(hidden_dim))
        self.uw = nn.Parameter(torch.zeros(hidden_dim))

    def attention(self,lstm_out,final):
        # implemented idea of attention from Yang 2016

        # find u_i for each word, u_i = tanh(W h_i +b)
        # print(lstm_out.shape)
        u = torch.tanh(torch.matmul(lstm_out,self.w_att))

        # print(u.shape)

        # find alpha_i using softmax on u_i
        ui_uw = torch.matmul(u,self.uw)
        # print(ui_uw.shape)

        alpha = F.softmax(ui_uw,1)
        # print(alpha.shape)

        # find document vector v = sum_i(alpha_i*h_i)
        # VECTORIZE THIS!!!
        # v=torch.zeros(lstm_out.shape[0],lstm_out.shape[2])
        # for b in range(0,lstm_out.shape[0]):
        #     for i in range(0,lstm_out.shape[1]):
        #         v[b,:]+=alpha[b,i]*lstm_out[b,i,:]
        # return v

        v=torch.matmul(lstm_out.transpose(1,2),alpha.unsqueeze(2)).squeeze(2)
        return v

        # print(str(lstm_out.shape) + ", " + str(self.w_att.shape))
        # ui = torch.tanh(torch.matmul(self.w_att,lstm_out) + self.b_att)
        # alphai = F.softmax(ui)
        # print(alphai.shape)
        # out = torch.matmul(alphai,lstm_out)

        # return out
        # hidden = final.squeeze(0)
        # attn = torch.bmm(lstm_out, hidden.unsqueeze(2)).squeeze(2)
        # soft_attn = F.softmax(attn, 1)
        # new_hidden = torch.bmm(lstm_out.transpose(1, 2), soft_attn.unsqueeze(2)).squeeze(2)
        #
        # return new_hidden

    def forward(self,tweet):

        batch_size = tweet.size(0)

        #embed the tweet
        embed = self.word_embed(tweet)
        #lstm layer
        lstm_out, (final_hidden, final_cell) = self.lstm(embed)

        attn_out = self.attention(lstm_out, final_hidden)

        #output connected layer
        outlayer = self.outlayer(attn_out)
        #softmax
        out = self.sig(outlayer)

        out = out.view(batch_size, -1)
        out = out[:,-1]
        return out
