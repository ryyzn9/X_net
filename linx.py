import torch 
from torch import nn
 # you write all the code by your self not any copy paser you should proud yourself
import math

from xpos_relative_position import XPOS
# seq = sequence length 
# d_model = size of embedding vector 
# h = number of heads\\



class LayerNorm(nn.Module):
    def __init__(self,features : int, eps: float = 10**-6)->None:
        
        super().__init__()
        self.eps = eps # esp is to prevent the division by zero when the std is very low
        self.alpha = nn.Parameter(torch.ones(features))# alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(features))# bias is also a learable parameter

    def forward(self,x):
        # x = (batch, seq_len, hidden_size)
        
        #keep the dimentino for broadcasting keepdim = True
        mean =  x.mean(dim =-1,  keepdim = True)#(batch_size,seq_len,1)
        #keep the dimentino for broadcasting keepdim = True

        std = x.std(dim = -1,keepdim = True) #(batch_size,seq_len,1)  
        
        return self.alpha* (x - mean )/(std +self.eps) + self.bias
    

class Feedforwaed(nn.Module):

    def __init__(self,d_model:int,d_ff:int,dropout:float)->None:
        super().__init__()  # Add this line

        self.L1 = nn.Linear(d_model,d_ff)
        self.L2 = nn.Linear(d_ff,d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        x = self.L1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.L2(x)
        return x
    



class InputEmbedding(nn.Module):
    def __init__(self, d_model:int,vocab_size:int)->None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self,x):
        return self.embedding(x)*math.sqrt(self.d_model)
    
#positional encoding
class PositioalEncoding(nn.Module):
    def __init__(self,d_model:int, seq_len:int,dropout:float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        #CREATE a matrixx of shape(seq_len ,d_model)
        pe = torch.zeros(seq_len,d_model)
        # create a vector of shape (seq_len,1)
        position = torch.arange(0,seq_len,dtype = torch.float).unsqueeze(1)#(seq_len,1)
        div_term = torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))#(d_model/2)
        #appply the sine to even positions
        pe[:,0::2] = torch.sin(position * div_term) #0 , 2,4,6,8 position
        #apply cos to the odd positions
        pe[:,1::2] = torch.cos(position*div_term)# 1,3,,5,7,9 position

        pe = pe.unsqueeze(0) #  (1, seq_len,d_model)
        # register the positional encoding as buffer
        self.register_buffer("pe",pe)
        #his ensures that the positional encoding is properly saved and loaded
        #  when you save and load instances of the model.
    def forward(self, x):
        x = x +(self.pe[:,:x.shape[1], :]).requires_grad_(False)#(batch,seq_len,d_model)
        return self.dropout(x)
    
class ResNet(nn.Module):
    def __init__(self, features:int , dropout : float)->None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(features)

    def forward(self,x,sublayer):
        return x + self.dropout(sublayer(self.norm(x))) 


class MultiheadAttn(nn.Module):
    def __init__(self,
                 d_model:int,
                 head_num : int ,
                 dropout:float)->None:

        super().__init__()
        self.d_model = d_model# (embedding vector)
        self.head_num = head_num

        assert d_model%head_num == 0, f"d_model is not divisible by h where h is {head_num} and d_model is {d_model}"
        self.d_k = d_model// head_num #dimention vector seen by each head

        self.w_q = nn.Linear(d_model,d_model,bias = False)
        self.w_k = nn.Linear(d_model,d_model,bias = False)
        self.w_v = nn.Linear(d_model,d_model,bias = False)
        self.w_o = nn.Linear(d_model,d_model,bias = False)
        self.w_g = nn.Linear(d_model,d_model,bias = False)
        self.group_norm = nn.GroupNorm(head_num, self.d_k)
        self.swish = lambda x: x * torch.sigmoid(x)

        self.dropout = nn.Dropout(dropout)
        self.gammas = (1 - torch.exp(torch.linspace(math.log(1/32), math.log(1/512), head_num))).detach().cpu().tolist()

        self.xpos = XPOS(self.d_model)
    @staticmethod
    def rention(D,query,
                  key,
                  value
                  ):



        Q = query
        K = key
        V = value

       # (n , d) @ (d ,k) --> (n,k)
        x=(Q @ K) * D.transpose(-2,-1).unsqueeze(0)

        
        x = x.softmax(dim=-1)
        

        ret = x  #(batch, head_num, seq_len, seq_len)
        #(n  k) @(d , k).T
        x=  ret @ V.transpose(-2,-1)
        
        
        return x



    def forward(self,q ,k ,v):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # query =self.xpos(query)
        # key = self.xpos(key)
        sequence_length = query.shape[1]

        x=q

        D = [self._get_D(self.new_method(sequence_length),self.d_k,gamma=gamma).to(device="cuda") for gamma in self.gammas]


       # Convert the list of PyTorch tensors to a single PyTorch tensor
        D= torch.stack(D)

       # Add an extra dimension at the beginning


        query =self.xpos(query)
        key =self.xpos(key)



        wx=self.get_EF(sequence_length,self.d_k).to(device="cuda")



        #break the q ,k,v for multilple heads
        #(batch,seq_len,d_model)-->(batch_size,seq_len,head_num,dk)-->(batch,head_num,seq_len,d_k)

        query = query.view(query.shape[0],query.shape[1],self.head_num,self.d_k).transpose(1,2)
        key =   key.view(key.shape[0],key.shape[1],self.head_num,self.d_k).transpose(1,2)
        value = value.view(value.shape[0],value.shape[1],self.head_num,self.d_k).transpose(1,2)

        key = wx(key.transpose(-2,-1))
        value = wx(value.transpose(-2,-1))


        y  = MultiheadAttn.rention(D,query,key ,value=value )
        #combine att the heads means concat
        # (batch ,h,seq_len,d_k)-->(batch,seq_len,head_num,dk)-->(batch,seq_len,d_model)
        y = y.transpose(1,2).contiguous().view(y.shape[0],-1,self.head_num*self.d_k)
        #(batch,seq_len,d_model)-->(batch,seq_len,d_model)

        # multiply by  W_o
        y_shape = y.shape
        y = self.group_norm(y.reshape(-1, self.d_k)).reshape(y_shape)

        return self.w_o(self.swish( self.w_g(x)*y ))


    def new_method(self, sequence_length):
        return sequence_length


    def _get_D(self, sequence_length, heads,gamma):
        n = torch.arange(heads).unsqueeze(1)
        m = torch.arange(sequence_length).unsqueeze(0)

        # Broadcast self.gamma ** (n - m) with appropriate masking to set values where n < m to 0
        D = (gamma ** (n - m)) * (n >= m).float()  #this results in some NaN when n is much larger than m
        # fill the NaN with 0
        D[D != D] = 0

        return D
    def get_EF(self,input_size, dim, method="learnable", head_dim=None, bias=True):
          """
          Retuns the E or F matrix, initialized via xavier initialization.
          This is the recommended way to do it according to the authors of the paper.
          Includes a method for convolution, as well as a method for no additional params.
          """
          assert method == "learnable" or method == "convolution" or method == "no_params", "The method flag needs to be either 'learnable', 'convolution', or 'no_params'!"
          if method == "convolution":
              conv = nn.Conv1d(head_dim, head_dim, kernel_size=int(input_size/dim), stride=int(input_size/dim))
              return conv
          if method == "no_params":
              mat = torch.zeros((input_size, dim))
              torch.nn.init.normal_(mat, mean=0.0, std=1/dim)
              return mat
          lin = nn.Linear(input_size, dim, bias)
          torch.nn.init.xavier_normal_(lin.weight)
          return lin

    

    
class EncoderBlock(nn.Module):
    def __init__(self,features:int,Multi_Head_attention_block:MultiheadAttn,Feedforwaedblock:Feedforwaed,dropout:float)->None:
        super().__init__()  # Add this line

        self.MHAttention = Multi_Head_attention_block
        self.feed_forward = Feedforwaedblock
        self.resnet = nn.ModuleList([ResNet(features,dropout) for _ in range(2)]) # need two resnet block
    def forward(self,x):
        x = self.resnet[0](x ,lambda x:self.MHAttention(x,x,x))
        x = self.resnet[1](x , self.feed_forward)
        return x
    
class Encoder(nn.Module):
    def __init__(self, features :int ,
                  
                 layers:nn.ModuleList) -> None:
        
        super().__init__()
        self.layer = layers
        self.norm = LayerNorm(features)
    def forward(self,x):
        for layer in self.layer:
            x = layer (x)
        return self.norm(x)
    

class DecoderBlock(nn.Module):
    def __init__(self,features:int,
                 MHAttention:MultiheadAttn,
                 crossattention:MultiheadAttn,
                 feed_forward:Feedforwaed,
                 drop:float
                 ) -> None:
        super().__init__()

        self.attention = MHAttention
        self.crossAttenion = crossattention
        self.FF = feed_forward
        self.resnet = nn.ModuleList([ResNet(features,drop) for _ in range(3)]) # need 3 resnet block  
    
    def forward(self,x,encoder_output):
        x = self.resnet[0](x,lambda x : self.attention(x,x,x))
        x = self.resnet[1](x, lambda x : self.crossAttenion(x,encoder_output,encoder_output)) 
        x = self.resnet[2](x,self.FF)
        return x
class Decoder(nn.Module):
    def __init__(self,features:int,layers:nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm(features)
    def forward(self,x,encoder_output)->None:
        for layer in self.layers:
            x = layer(x,encoder_output)
        return self.norm(x)
    
    
class projectionLayer(nn.Module): # the last output layer
    def __init__(self, d_model,vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model,vocab_size)
    def forward(self,x)->None:
        #(batch,seq_len,d_model)-->(batch,seq_len,vocab_size)
        return self.proj(x)


class Transformer(nn.Module):
    def __init__(self,encoder:Encoder,
                 decoder:Decoder,
                 src_embed:InputEmbedding,
                 tgt_embed:InputEmbedding,
                 
                 projection_layer:projectionLayer)->None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        
        self.proj = projection_layer
    def encode (self,src):
        #(batch,seq_len,d_model)
        src = self.src_embed(src)
        return self.encoder(src)
    

    def decode (self,encoder_output:torch.Tensor,
                
                tgt:torch.Tensor,
                ):
        
        tgt = self.tgt_embed(tgt)
        return self.decoder(tgt,encoder_output)
    def project(self,x):
        return self.proj(x)
    
        

def build_transformer(en_vocab_size:int,
          tgt_vocab_size:int,
          src_seq_len:int,
          tgt_seq_len:int,
          d_model:int=512,
          Num_block:int=2,
          head:int=4,
          dropout:float=0.1,
          d_ff:int=248)->Transformer:
    # vreate the embeddding layers
    en_embed = InputEmbedding(d_model,en_vocab_size)
    de_embed = InputEmbedding(d_model, tgt_vocab_size)
    #create the positional encoding layers

    


    #vreate the encodre blocks
    encoder_blocks =[]
    for _ in range(Num_block):
        en_selfattention = MultiheadAttn(d_model, head,dropout)
        
        FF = Feedforwaed(d_model,
                       d_ff,
                       dropout)
        encoder_b = EncoderBlock(d_model,en_selfattention,FF,dropout)
        encoder_blocks.append(encoder_b)

    decoder_blocks =[]
    for _ in range(Num_block):
        de_selfattention = MultiheadAttn(d_model, head,dropout)
        de_crossAttention = MultiheadAttn(d_model, head,dropout)
        FF = Feedforwaed(d_model,
                       d_ff,
                       dropout)   
        decoder_b = DecoderBlock(d_model,de_selfattention,de_crossAttention,FF,dropout) 
        decoder_blocks.append(decoder_b)

    # create the encoder and decoder    
    encoder = Encoder(d_model,nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model,nn.ModuleList(decoder_blocks))

    projection_layer = projectionLayer(d_model,
                                       tgt_vocab_size)
    
    transformer1 = Transformer(encoder,decoder,en_embed,de_embed,projection_layer)  

    # initialize the parameters
    for p in transformer1.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform_(p)
    return transformer1

        








        






