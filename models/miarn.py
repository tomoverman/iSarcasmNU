import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class MIARN(nn.Module):
    
    def __init__(self, embed_dim, hidden_dim, vocab_size,seq_len):
        super(MIARN, self).__init__()
        #initialize some variables
        self.hidden_dim=hidden_dim
        self.embed_dim=embed_dim
        self.vocab_size=vocab_size
        self.seq_len=seq_len

        #form the layers of the network
        self.word_embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, batch_first=True)
        #predictions are either sarcasm or not, so two labels
        self.projlayer = nn.Linear(hidden_dim+embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.outlayer = nn.Linear(hidden_dim, 1)
        self.sig = nn.Sigmoid()

        k=embed_dim
        self.intra_att_1 = nn.Linear(2*embed_dim,k)
        self.intra_att_2 = nn.Linear(k,1)

        self.idxs = torch.cartesian_prod(torch.tensor(range(0, seq_len)), torch.tensor(range(0, seq_len)))

    def intra_attention(self,embed):
        # VECTORIZE THIS
        # s=torch.zeros(embed.shape[0],self.seq_len,self.seq_len)
        # for i in range(0,self.seq_len):
        #     for j in range(0,self.seq_len):
        #         concat = torch.cat((embed[:,i,:],embed[:,j,:]),1)
        #         # keep i=j at 0 because the intra-word attention should not represent the same word
        #         if not i==j:
        #             s[:,i,j] = torch.matmul(concat,self.Wa) + self.ba
        # return s

        pair_concats = torch.cat([embed[:,self.idxs,:]], dim=1).view(embed.shape[0], -1, 2*embed.shape[2])
        s1 = self.intra_att_1(pair_concats)
        s2 = self.relu(s1)
        s3 = self.intra_att_2(s2)

        s3 = s3.view(embed.shape[0],self.seq_len,self.seq_len)
        # keep i=j at 0 because the intra-word attention should not represent the same word
        for i in range(0,self.seq_len):
            s3[:,i,i]=0.
        # print(torch.allclose(s,s1,atol=10**-3))
        return s3

    def forward(self,tweet):
        batch_size = tweet.size(0)
        embed = self.word_embed(tweet)

        # intra-attention layer
        s = self.intra_attention(embed)
        a = F.softmax(torch.max(s,1)[0],0)

        # VECTORIZE THIS
        # v = torch.zeros(embed.shape[0],embed.shape[2])
        # for b in range(0,embed.shape[0]):
        #     for i in range(0,self.seq_len):
        #         v[b,:] += a[b,i]*embed[b,i,:]

        v = torch.matmul(embed.transpose(1,2),a.unsqueeze(2)).squeeze(2)

        # LSTM layer
        lstm_out, (final_hidden, final_cell) = self.lstm(embed)

        # combine intra-attention and LSTM

        combined = torch.cat((v,final_hidden.squeeze(0)),1)

        # nonlinear projection layer
        projlayer = self.projlayer(combined)
        relu_proj = self.relu(projlayer)
        # output connected layer
        outlayer = self.outlayer(relu_proj)
        # softmax
        out = self.sig(outlayer)

        out = out.view(batch_size, -1)
        out = out[:, -1]

        return out
