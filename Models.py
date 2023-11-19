import torch
import torch.nn.functional as F
from torch_geometric.nn import APPNP
from torch import nn
import math
from performer_pytorch import FastAttention


# Model construction
class APPNPTransformerBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        #assert embedding_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."
        self.embedding_dim = args.hidden
        self.num_heads = args.n_heads
        self.head_dim = self.embedding_dim // self.num_heads

        # for convolution
        self.lin1 = nn.Linear(args.num_features, self.embedding_dim) 
        self.lin2 = nn.Linear(2*self.embedding_dim, args.num_classes)
        self.prop1 = APPNP(10, 0.1)

        # for MHA
        self.qkv_proj = nn.Linear(args.num_features, 3*self.embedding_dim)
        self.o_proj = nn.Linear(self.embedding_dim, args.num_classes)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop1.reset_parameters()
        self.qkv_proj.reset_parameters()
        self.o_proj.reset_parameters()

    def qkv_calculation(self, unsqueezed):
        batch_size, seq_length, embedd_dim = unsqueezed.size()
        qkv = self.qkv_proj(unsqueezed)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)
        d_k = q.size()[-1]
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(d_k)
        attention = F.softmax(attn_logits, dim=-1)
        values = torch.matmul(attention, v)

        # Determine value outputs
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embedding_dim)

        return values

    def forward(self, data, edge_index):
        unsqueezed = data.unsqueeze(0)
        squeezed = data
        # get the values
        transembedding = self.qkv_calculation(unsqueezed)
       
        # embed of conv 
        squeezed = F.dropout(squeezed, training=self.training)
        gnnembedding1 = self.lin1(squeezed)     
        gnnembedding_bein = torch.concat((gnnembedding1,transembedding.squeeze(0)), 1)
        gnnembedding_bein = F.elu(gnnembedding_bein)
        gnnembedding_bein = F.dropout(gnnembedding_bein, training=self.training)
        gnnembedding2 = self.lin2(gnnembedding_bein)
        gnnembedding2 = F.elu(gnnembedding2)
        gnnembedding2 = F.dropout(gnnembedding2, training=self.training)
        gnnembeddingfinal = self.prop1(gnnembedding2, edge_index)

        return F.log_softmax(gnnembeddingfinal, dim=1), gnnembeddingfinal




# Model construction
class fastAPPNPTransformerBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        #assert embedding_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."
        self.embedding_dim = args.hidden
        self.num_features = args.num_features
        self.num_heads = args.n_heads
        self.head_dim = self.embedding_dim // self.num_heads

        # for convolution
        self.lin1 = nn.Linear(args.num_features, self.embedding_dim) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.lin2 = nn.Linear(2*self.embedding_dim, args.num_classes)
        self.prop1 = APPNP(10, 0.1)

        # for MHA
        self.q_proj = nn.Linear(args.num_features, self.embedding_dim)
        self.k_proj = nn.Linear(args.num_features, self.embedding_dim)
        self.v_proj = nn.Linear(args.num_features, self.embedding_dim)
        self.fastattn = FastAttention(dim_heads = self.head_dim, causal = False)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.q_proj.reset_parameters()
        self.k_proj.reset_parameters()
        self.v_proj.reset_parameters()
        self.prop1.reset_parameters()

    def qkv_calculation(self, unsqueezed):
        batch_size, seq_length, embedd_dim = unsqueezed.size()
        q = self.q_proj(unsqueezed)
        k = self.k_proj(unsqueezed)
        v = self.v_proj(unsqueezed)

        q = q.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        k = k.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        k = k.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        v = v.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        v = v.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]

        values = self.fastattn(q, k, v)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embedding_dim)

        return values

    def forward(self, data, edge_index):
        data = F.dropout(data, training=self.training)

        unsqueezed = data.unsqueeze(0)
        squeezed = data
        transembedding = self.qkv_calculation(unsqueezed)
        # embed of conv 
        gnnembedding1 = self.lin1(squeezed)     
        gnnembedding_bein = torch.concat((gnnembedding1,transembedding.squeeze(0)), 1)
        gnnembedding_bein = F.elu(gnnembedding_bein)
        gnnembedding_bein = F.dropout(gnnembedding_bein, training=self.training)
        gnnembedding2 = self.lin2(gnnembedding_bein)
        gnnembedding2 = F.elu(gnnembedding2)
        gnnembedding2 = F.dropout(gnnembedding2, training=self.training)
        gnnembeddingfinal = self.prop1(gnnembedding2, edge_index)

        return F.log_softmax(gnnembeddingfinal, dim=1), gnnembeddingfinal
