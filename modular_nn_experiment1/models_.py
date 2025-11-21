from einops import rearrange, einsum
import einx
import math
from jaxtyping import Float, Bool, Int,Array
from typing import *
from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class Extractor(nn.Module, ABC):
    def __init__(self, d_in: int, d_out: int, d_hidden: int, img_size: int):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.d_hidden = d_hidden
        self.img_size = img_size

    @abstractmethod
    def forward(self, x: Float[Array, "batch channel height width"]) -> Float[Array, "batch seq_len dim"]:
        pass


class Dense(nn.Module,ABC):
    def __init__(self):
        super().__init__()
        
    @staticmethod
    
    def trunc_normal_init(tensor: Float[Array, "..."], d_in: int, d_out: int):
        std=math.sqrt(2.0/ (d_in + d_out))
        nn.init.trunc_normal_(tensor, std=std, a=-3*std, b=3*std)
    
    @abstractmethod
    def forward(self,x:Float[Array, "... d_in"]) -> Float[Array, "... d_out"]:
        pass
    

class Linear(Dense):
    def __init__(self, d_in: int, d_out: int,bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty( d_out, d_in, requires_grad=True))
        self.trunc_normal_init(self.weight,d_out, d_in)
        if bias:
            self.bias = nn.Parameter(torch.zeros(d_out, requires_grad=True))
        else:
            self.bias = None
    
    def forward(self, x: Float[Array, "... d_in"]) -> Float[Array, "... d_out"]:
         return einsum(x,self.weight,"... d_in, d_out d_in-> ... d_out")+ (self.bias if self.bias is not None else 0)
     
class Conv2d(Dense):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size, requires_grad=True))
        self.trunc_normal_init(self.weight, in_channels * kernel_size * kernel_size, out_channels)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels, requires_grad=True))
        else:
            self.bias = None
    
    def forward(self, x: Float[Array, "batch in_channels height width"]) -> Float[Array, "batch out_channels h w"]:
        return nn.functional.conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding)




    
class ResidualLayer(nn.Module):
    def __init__(self, channels_in:int):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.SiLU(),
            nn.BatchNorm2d(channels_in),
            nn.Conv2d(channels_in, channels_in, kernel_size=3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.SiLU(),
            nn.BatchNorm2d(channels_in),
            nn.Conv2d(channels_in, channels_in, kernel_size=3, padding=1)
        
        )
        
    def forward(self, x: Float[Array, "batch channels height width"]) -> Float[Array, "batch channels height width"]:
        res = x
        x= self.block1(x)
        x= self.block2(x)
        
        x+=res
        return x


class MiniUNet(Extractor):
    def __init__(self, d_in=1, d_out=8, d_hidden=4, kernel_num=4, img_size=32):
        super().__init__(d_in, d_out, d_hidden, img_size)
        self.init_conv = nn.Sequential(Conv2d(d_in, d_hidden, kernel_num, padding=1), 
                                       nn.BatchNorm2d(d_hidden), 
                                       nn.SiLU())
        
        self.res_blocks=ResidualLayer(d_hidden)
        self.conv2=Conv2d(d_hidden, d_out,kernel_size=3, stride=2, padding=1)
    def forward(self, x: Float[Array, "batch channel height width"]) -> Float[Array, "batch seq_len dim"]:
        x= self.init_conv(x)
        res=x
        x = self.res_blocks(x)
        x = self.conv2(x)
        x= rearrange(x, "b c h w -> b (h w) c")
        return x



def scale_dot_product_attetnion(
    Query:Float[Array,"... queries d_k"],
    Key:Float[Array,"... keys d_k"],
    Value:Float[Array,"... values d_v"],
    mask:Bool[Array,"... queries keys"] | None = None,
)->Float[Array,"... queries d_v"]:
    d_k=Key.shape[-1]
    attention_score=einsum(Query,Key,"... queries d_k,... keys d_k-> ... queries keys")/math.sqrt(d_k)
    if mask is not None:
        attention_score=torch.where(mask, attention_score, torch.tensor(float('-inf')))
    attention_weights=torch.softmax(attention_score,dim=-1)
    
    return einsum(attention_weights,Value,"... query key, ... key d_v ->  ... query d_v")

class Embedding(nn.Module):
    def __init__(self,num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(nn.init.trunc_normal_(torch.empty(num_embeddings, embedding_dim), std=1.0,
                                 a=-3,b=3))
        
    def forward(self,token_ids:Int[Array,"..."])->Float[Array,"... d_model"]:
        token_ids = token_ids.long() 
        return self.weight[token_ids,:]
class RMSNorm(nn.Module):
    def __init__(self,d_model:int,eps:float=1e-5,device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
    
    def forward(self,x:Float[Array,"batch_size seq_len d_model"])->Float[Array,"batch_size ..."]:
        in_dtype=  x.dtype
        x=x.to(torch.float32)
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        
        return self.weight * (x *rms)
class RotaryEmbedding(nn.Module):
    def __init__(self, d_k: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        self.register_buffer(
            "_freq_cis_cache",
            RotaryEmbedding._init_cache(max_seq_len, d_k, theta), persistent=False
        )
    @staticmethod
    def _init_cache(max_seq_len:int, d_k:int, theta:float)-> Float[Array,"2 max_seq_len d_k/2"]:
        assert d_k % 2 == 0, "d_k must be even"
        d=torch.arange(0,d_k,2).float()
        t=torch.arange(0,max_seq_len).float()
        freqs= theta ** (-d / d_k)
        freqs= einsum(t,freqs,'t,f -> t f')
        cos,sin= torch.cos(freqs), torch.sin(freqs)
        return torch.stack((cos,sin))
    
    def forward(self,x:Float[Array,"... seq  d_k"],pos_ids: Int[Array, " ... seq"])->Float[Array,"...  seq d_k"]:
        x1, x2 = rearrange(x, "... (d r) -> ... d r", r=2).unbind(-1)
        cos, sin = einx.get_at('cos_sin [pos] half_dim, ... -> cos_sin ... half_dim', self._freq_cis_cache, pos_ids)
        x1_rot = cos * x1 - sin * x2
        x2_rot = sin * x1 + cos * x2
        result = einx.rearrange('... x_half, ... x_half -> ... (x_half (1 + 1))', x1_rot, x2_rot).contiguous()
        return result

class CustomizedAttention(nn.Module,ABC):
   def __init__(
       self,
       d_model:int,
       num_heads:int,
       positional_encoder:None,
   ):
       super().__init__()
       assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
       self.d_model = d_model
       self.num_heads = num_heads
       self.d_k = d_model // num_heads
       self.d_v=self.d_k
       
       self.q_proj=Linear(self.d_model,self.num_heads*self.d_k)
       self.k_proj=Linear(self.d_model,self.num_heads*self.d_k)
       self.v_proj=Linear(self.d_model,self.num_heads*self.d_v)
       
       self.out_proj=Linear(self.num_heads*self.d_v,self.d_model)
       
       self.positional_encoder = positional_encoder
       
   @abstractmethod
   def forward(
        self,
        x:Float[Array,"... sequence d_k"],
        token_positions:Int[Array,"... seq"] | None = None)->Float[Array,"... sequence d_v"]:
        pass 
    


class CausalMultiHeadAttention(CustomizedAttention):
    def __init__(self, d_model: int, num_heads: int, positional_encoder:  RotaryEmbedding):
        super().__init__(d_model, num_heads, positional_encoder)
    
    def forward(
        self,
        x: Float[Array, "... sequence d_k"],
        token_positions: Int[Array, "... seq"] | None = None
    ) -> Float[Array, "... sequence d_v"]:
        
        *b,seq_len,d_model=x.shape
        
        assert d_model ==self.d_model
        Q=self.q_proj(x)
        K=self.k_proj(x)
        V=self.v_proj(x)
        
        Q,K,V=(
            rearrange(X,"... sequence (heads d_k)->... heads sequence d_k" ,heads=self.num_heads)
            for X in (Q, K, V)
        )
       
        if token_positions is  None:
            token_positions = einx.rearrange("seq -> b... seq",torch.arange(seq_len, device=x.device),b=[1] * len(b))
        token_positions=rearrange(token_positions, "... seq -> ... 1 seq ")
        Q=self.positional_encoder(Q,token_positions)
        K=self.positional_encoder(K,token_positions)
        
        #seq=torch.arange(seq_len, device=x.device)
        #qi=einx.rearrange("query ->b... 1 query 1",seq,b=[1]*len(b))
        #ki=einx.rearrange("key -> b... 1  1 key ",seq,b=[1]*len(b))
        #casual_mask=qi >=ki
        
        attn_output= scale_dot_product_attetnion(
            Q, K, V, mask=None
        )
        
        attn_output=rearrange(attn_output,"... h seq d_v -> ... seq (h d_v)").contiguous()
        output=self.out_proj(attn_output)
        return output
    
    
class MiniVisionTransformer(Extractor):
    def __init__(self, d_in=1, d_out=8, d_hidden=16, patch_size=4,num_heads=2,img_size=32):
        super().__init__(d_in, d_out, d_hidden, img_size)
        self.initconv=Conv2d(d_in,d_hidden, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_hidden))
        self.positions= Embedding((img_size // patch_size) ** 2 + 1, d_hidden)
        self.position_encoder = RotaryEmbedding(
    d_k=d_hidden // num_heads,
    max_seq_len=(img_size // patch_size) ** 2 + 1
)
 
        self.attn= CausalMultiHeadAttention(
            d_model=d_hidden,
            num_heads=num_heads,
            positional_encoder=self.position_encoder,
        )
        self.norm1 = RMSNorm(d_hidden)
        self.norm2 = RMSNorm(d_hidden)
        self.ffn = nn.Sequential(
            Linear(d_hidden, d_hidden * 4),
            nn.GELU(),
            Linear(d_hidden * 4, d_hidden)
        )
        self.proj= Linear(d_hidden, d_out)
        
    def forward(self, x: Float[Array, "batch channel height width"]) -> Float[Array, "batch seq_len dim"]:
        x= self.initconv(x)
        b, c, h, w = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")
        cls_token = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        pos_ids=torch.arange(x.size(1), device=x.device)
        pos_ids=rearrange(pos_ids,'seq -> 1 seq')
        x+=self.positions(pos_ids)
        x_attn=self.attn(self.norm1(x))
        x_sub=x+x_attn
        x_ffn=self.ffn(self.norm2(x_sub))
        ffn_sub=x_sub+x_ffn
        
        return  self.proj(ffn_sub)
    
    
    


class MLPExtractor(Extractor):
    def __init__(self, d_in=32*32, d_out=8, d_hidden=64, img_size=32):
        super().__init__(d_in, d_out, d_hidden, img_size)
        self.layer = nn.Sequential(
            Linear(d_in, d_hidden),
            nn.ReLU(),
            Linear(d_hidden, d_out)
        )

    def forward(self, x: Float[Array, "batch channel height width"]) -> Float[Array, "batch seq_len dim"]:
        
        #x = x.view(x.size(0), -1)  # (b, d_in)
        #x = self.layer(x).unsqueeze(1)  # (b, 1, d_out)
        x=rearrange(x, 'b c h w -> b (c h w)')
        x= self.layer(x)
        x= rearrange(x, 'b d_out -> b 1 d_out')  # Reshape to (b, 1, d_out)
      
        
        return x
    
    
    
class PCAMLP(Extractor):
    def __init__(self, d_in=32*32, d_out=8, d_hidden=32, img_size=32, n_components=64):
        super().__init__(d_in, d_out, d_hidden, img_size)
        self.n_components = n_components
        self.fitted = False
        self.register_buffer("pca_components", torch.empty(self.n_components, d_in))
        self.register_buffer("mean", torch.empty(d_in))  # [d_in]

        self.linear = nn.Sequential(
            Linear(n_components, d_hidden),
            nn.ReLU(),
            Linear(d_hidden, d_out)
        )

    def _fit_pca_from_batch(self, x: torch.Tensor):
       x_flat = x.view(x.size(0), -1)  # [batch, d_in]
       mean = x_flat.mean(dim=0)
       x_centered = x_flat - mean

 
       U, S, Vh = torch.linalg.svd(x_centered, full_matrices=False)
       available_components = Vh.size(0)  # ≤ batch_size

 
       used_components = min(self.n_components, available_components)

 
       self.pca_components[:used_components] = Vh[:used_components]
       if used_components < self.n_components:
          self.pca_components[used_components:] = 0.0  # padding with zeros

       self.mean[:] = mean
       self.fitted = True

    def forward(
        self,
        x: Float[Array, "batch channel height width"]
    ) -> Float[Array, "batch seq_len dim"]:
        x = x.view(x.size(0), -1)  # [batch, d_in]

        if not self.fitted:
        
            self._fit_pca_from_batch(x.detach())

        x_centered = x - self.mean  # [batch, d_in]
        x_pca = torch.matmul(x_centered, self.pca_components.T)  # [batch, n_components]
        out = self.linear(x_pca)  # [batch, d_out]
        return out.unsqueeze(1)  # [batch, 1, d_out]



def silu(x: Float[Array,"... d_model"])->Float[Array,"... d_model"]:
    return x * torch.sigmoid(x)
class SwiGLU(nn.Module):
    def __init__(self,d_model:int,d_ff:int):
        super().__init__()
        self.w1= Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3= Linear(d_model, d_ff)

    def forward(self,x:Float[Array,"... d_model"])->Float[Array,"... d_ff"]:

        return self.w2(silu(self.w1(x)* self.w3(x)))



class GeluGateProjector(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.activation=nn.functional.gelu

    def forward(self, x: Float[Array, "... d_model"]) -> Float[Array, "... d_model"]:
        proj= self.w1(x)
        return self.activation(proj) * proj


class TopKSelector(nn.Module):
    def __init__(self, d_model:int,k:int):
        super().__init__()
        self.k = k
        self.score_net=nn.Sequential(
            Linear(d_model, d_model),
            SwiGLU(d_model, d_model)   
        ) 
        self.projector =GeluGateProjector(d_model,k)
    def forward(self,x:Float[Array,"... d_model"])->Float[Array,"... k"]:
        scores = self.score_net(x)
        topk_scores, topk_indices = torch.topk(scores, self.k, dim=-1)
        
        mask = torch.zeros_like(scores)
        mask.scatter_(dim=-1, index=topk_indices, value=1.0)
        x_masked = x * mask
        
        proj= self.projector(x_masked)
        return proj 



class UniversalClassifier(nn.Module):
    def __init__(self, d_in:int,d_out:int):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in,d_out)
        )

    def forward(self, x: Float[Array, "batch seq_len dim"]) -> Float[Array, "batch num_classes"]:
        x = x.mean(dim=1)  
        return self.classifier(x)


class FeatureEncoder(nn.Module):
    def __init__(self, extractor: Extractor, selector: TopKSelector):
        super().__init__()
        self.extractor = extractor
        self.selector = selector

    def forward(self, x: Float[Array, "batch channel height width"]) -> Float[Array, "batch d_selected"]:
        feats = self.extractor(x)            # (B, Seq, D)
        selected = self.selector(feats)      # (B, k) or (B, d_selected)
        return selected
    

    
    


