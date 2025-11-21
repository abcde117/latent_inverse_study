
from einops import rearrange, einsum
import einx
import math
from jaxtyping import Float, Bool, Int,Array
from typing import *
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from models_ import *
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

class Sampleable(ABC):

    @abstractmethod
    def sample(self, num_samples: int) -> Tuple[Float[Array, "batch ..."], Optional[Float[Array, "batch label_dim"]]]:
        pass
   
class IsotropicGaussian(nn.Module, Sampleable):
    def __init__(self, shape: List[int], std: float = 1.0):

        super().__init__()
        self.shape = shape
        self.std = std
        self.dummy = nn.Buffer(torch.zeros(1)) 
        
    def sample(self, num_samples:int) -> Tuple[Float[Array, "num_samples *shape"], Optional[None]]:
        return self.std * torch.randn(num_samples, *self.shape).to(self.dummy.device), None


class Alpha(ABC):
    def __init__(self):
        # Check alpha_t(0) = 0
        assert torch.allclose(
            self(torch.zeros(1,1,1,1)), torch.zeros(1,1,1,1)
        )
        # Check alpha_1 = 1
        assert torch.allclose(
            self(torch.ones(1,1,1,1)), torch.ones(1,1,1,1)
        )
        
    @abstractmethod
    def __call__(self, t: Float[Array,'num_samples 1 1 1 ']) -> Float[Array, 'num_samples 1 1 1']:
        """
        Evaluates alpha_t. Should satisfy: self(0.0) = 0.0, self(1.0) = 1.0.
        Args:
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - alpha_t (num_samples, 1, 1, 1)
        """ 
        pass

    def dt(self, t: Float[Array,'num_samples  1 1 1']) -> Float[Array, 'num_samples 1 1 1']:
        """
        Evaluates d/dt alpha_t.
        Args:
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - d/dt alpha_t (num_samples, 1, 1, 1)
        """ 
        #t = t.unsqueeze(1)
        t= rearrange(t, 'b 1 1 1 -> b 1 1 1')
        dt = vmap(jacrev(self))(t)
        #dt.view(-1, 1, 1, 1)
        dt= rearrange(dt, 'b ... -> b 1 1 1')
        return dt
    

class LinearAlpha(Alpha):
    
    def __call__(self, t:  Float[Array,'num_samples 1 1 1']) ->  Float[Array,'num_samples 1 1 1']:
        return t

    def dt(self, t:  Float[Array,'num_samples 1 1 1']) ->  Float[Array,'num_samples 1 1 1']:
        """
        Evaluates d/dt alpha_t.
        Args:
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - d/dt alpha_t (num_samples, 1, 1, 1)
        """ 
        return torch.ones_like(t)
    

class Beta(ABC):
    def __init__(self):
        # Check beta_0 = 1
        assert torch.allclose(
            self(torch.zeros(1,1,1,1)), torch.ones(1,1,1,1)
        )
        # Check beta_1 = 0
        assert torch.allclose(
            self(torch.ones(1,1,1,1)), torch.zeros(1,1,1,1)
        )
        
    @abstractmethod
    def __call__(self, t: Float[Array,'num_samples 1 1 1']) -> Float[Array, 'num_samples 1 1 1']:
        """
        Evaluates alpha_t. Should satisfy: self(0.0) = 1.0, self(1.0) = 0.0.
        Args:
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - beta_t (num_samples, 1, 1, 1)
        """ 
        pass 

    def dt(self, t: Float[Array,'num_samples 1 1 1']) -> Float[Array,'num_samples 1 1 1']:
        """
        Evaluates d/dt beta_t.
        Args:
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - d/dt beta_t (num_samples, 1, 1, 1)
        """ 
        t= rearrange(t, 'b 1 1 1 -> b 1 1 1')
        dt = vmap(jacrev(self))(t)
        #dt.view(-1, 1, 1, 1)
        dt= rearrange(dt, 'b ... -> b 1 1 1')
        return dt
    
class LinearBeta(Beta):
    def __call__(self, t: Float[Array,'num_samples 1 1 1']) -> Float[Array,'num_samples 1 1 1']:

        return 1-t
        
    def dt(self, t: Float[Array,'num_samples 1 1 1']) -> Float[Array,'num_samples 1 1 1']:
        return - torch.ones_like(t)
    
    
class ConditionalProbabilityPath(nn.Module, ABC):
    def __init__(self, p_simple: Sampleable, p_data: Sampleable):
        super().__init__()
        self.p_simple = p_simple
        self.p_data = p_data

    def sample_marginal_path(self, t: Float[Array,"num_sample 1 1 1"]) -> Float[Array, "num_sample c h w"]:
        """
        Samples from the marginal distribution p_t(x) = p_t(x|z) p(z)
        Args:
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - x: samples from p_t(x), (num_samples, c, h, w)
        """
        num_samples = t.shape[0]
        # Sample conditioning variable z ~ p(z)
        z, _ = self.sample_conditioning_variable(num_samples) # (num_samples, c, h, w)
        # Sample conditional probability path x ~ p_t(x|z)
        x = self.sample_conditional_path(z, t) # (num_samples, c, h, w)
        return x
    @abstractmethod
    def sample_conditioning_variable(self, num_samples: int) -> Tuple[
        Float[Array, "num_samples c h w"],  # z
        Float[Array, "num_samples label_dim"]  # y
    ]:
        """
        Samples the conditioning variable z and label y
        """
        pass

    @abstractmethod
    def sample_conditional_path(
        self,
        z: Float[Array, "num_samples c h w"],
        t: Float[Array, "num_samples 1 1 1"]
    ) -> Float[Array, "num_samples c h w"]:
        """
        Samples from the conditional distribution p_t(x|z)
        """
        pass

    @abstractmethod
    def conditional_vector_field(
        self,
        x: Float[Array, "num_samples c h w"],
        z: Float[Array, "num_samples c h w"],
        t: Float[Array, "num_samples 1 1 1"]
    ) -> Float[Array, "num_samples c h w"]:
        """
        Evaluates the conditional vector field u_t(x|z)
        """
        pass

    @abstractmethod
    def conditional_score(
        self,
        x: Float[Array, "num_samples c h w"],
        z: Float[Array, "num_samples c h w"],
        t: Float[Array, "num_samples 1 1 1"]
    ) -> Float[Array, "num_samples c h w"]:
        """
        Evaluates the conditional score of p_t(x|z)
        """
        pass
    
    
    
    
    
class GaussianConditionalProbabilityPath(ConditionalProbabilityPath):
    def __init__(self, p_data: Sampleable, p_simple_shape: List[int], alpha: Alpha, beta: Beta):
        p_simple = IsotropicGaussian(shape = p_simple_shape, std=1.0)
        super().__init__(p_simple, p_data)
        self.alpha = alpha
        self.beta = beta
        
    def sample_conditioning_variable(self, num_samples: int) -> Tuple[
        Float[Array, "num_samples c h w"],  # z
        Float[Array, "num_samples label_dim"]  ]:# y
        return self.p_data.sample(num_samples)
    
    def sample_conditional_path(self, z: Float[Array,"num_samples c  h w"], t: Float[Array,"num_samples  1  1  1"]) -> Float[
        Array,"num_samples  c  h  w"]:
    
        return self.alpha(t) * z + self.beta(t) * torch.randn_like(z)
    
    def conditional_vector_field(self, 
                                 x: Float[Array,"num_samples  c  h w"], 
                                 z:Float[Array,"num_samples  c h w"], 
                                 t: Float[Array,"num_samples  1 1  1"]) -> Float[Array,"num_samples  c  h w"]:
    
        alpha_t = self.alpha(t) # (num_samples, 1, 1, 1)
        beta_t = self.beta(t) # (num_samples, 1, 1, 1)
        dt_alpha_t = self.alpha.dt(t) # (num_samples, 1, 1, 1)
        dt_beta_t = self.beta.dt(t) # (num_samples, 1, 1, 1)

        return (dt_alpha_t - dt_beta_t / beta_t * alpha_t) * z + dt_beta_t / beta_t * x
    
    
    def conditional_score(self, 
                            x: Float[Array,"num_samples  c h  w"], 
                                 z:Float[Array,"num_samples  c  h w"], 
                                 t: Float[Array,"num_samples 1  1  1"]) -> Float[Array,"num_samples  c  h  w"]:
        alpha_t = self.alpha(t)
        beta_t = self.beta(t)
        return (z * alpha_t - x) / beta_t ** 2 + 1e-4

class ODE(ABC):
    @abstractmethod
    def drift_coefficient(
        self,
        xt: Float[Array, "bs c h w"],
        t: Float[Array, "bs 1"],
        **kwargs
    ) -> Float[Array, "bs c h w"]:
     
        pass

class SDE(ABC):
    @abstractmethod
    def drift_coefficient(
        self,
        xt: Float[Array, "bs c h w"],
        t: Float[Array, "bs 1 1 1"],
        **kwargs
    ) -> Float[Array, "bs c h w"]:
    
        pass

    @abstractmethod
    def diffusion_coefficient(
        self,
        xt: Float[Array, "bs c h w"],
        t: Float[Array, "bs 1 1 1"],
        **kwargs
    ) -> Float[Array, "bs c h w"]:
    
        pass
    

class Simulator(ABC):
    @abstractmethod
    def step(
        self, 
        xt: Float[Array, "bs c h w"],
        t: Float[Array, "bs 1 1 1"],
        dt: Float[Array, "bs 1 1 1"],
        **kwargs)-> Float[Array, "bs c h w"]:
        pass

    @torch.no_grad()
    def simulate(self, 
                 x: Float[Array, "bs c h w"],
                 ts: Float[Array, "bs nts 1 1 1"], 
                 **kwargs)-> Float[Array, "bs c h w"]:
        nts = ts.shape[1]
        for t_idx in tqdm(range(nts - 1)):
            t = ts[:, t_idx]
            h = ts[:, t_idx + 1] - ts[:, t_idx]
            x = self.step(x, t, h, **kwargs)
        return x
    
    @torch.no_grad()
    def simulate_with_trajectory(self, 
                                 x: Float[Array, "bs c h w"], 
                                 ts: Float[Array, "bs nts 1 1 1"], 
                                 **kwargs)-> Float[Array, "bs nts c h w"]:
        xs = [x.clone()]
        nts = ts.shape[1]
        for t_idx in tqdm(range(nts - 1)):
            t = ts[:,t_idx]
            h = ts[:, t_idx + 1] - ts[:, t_idx]
            x = self.step(x, t, h, **kwargs)
            xs.append(x.clone())
        return torch.stack(xs, dim=1)

    
class EulerSimulator(Simulator):
    def __init__(self, ode: ODE):
        self.ode = ode
        
    def step(self, 
             xt: torch.Tensor, 
             t: torch.Tensor,
             h: torch.Tensor, **kwargs):
        return xt + self.ode.drift_coefficient(xt,t, **kwargs) * h
    
class EulerMaruyamaSimulator(Simulator):
    def __init__(self, sde: SDE):
        self.sde = sde
        
    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor, **kwargs):
        return xt + self.sde.drift_coefficient(xt,t, **kwargs) * h + self.sde.diffusion_coefficient(xt,t, **kwargs) * torch.sqrt(h) * torch.randn_like(xt)
    


class MNISTSampler(nn.Module, Sampleable):
    def __init__(self):
        super().__init__()
        self.dataset = datasets.MNIST(
            root="./data",
            train=True,
            download=False,
            transform=transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])
        )
        self.dummy = nn.Buffer(torch.zeros(1)) # Will automatically be moved when self.to(...) is called...

    def sample(self, num_samples: int) -> Tuple[Float[Array, "batch ..."], Optional[Float[Array, "batch label_dim"]]]:
        if num_samples > len(self.dataset):
            raise ValueError(f"num_samples exceeds dataset size: {len(self.dataset)}")
        
        indices = torch.randperm(len(self.dataset))[:num_samples]
        samples, labels = zip(*[self.dataset[i] for i in indices])
        samples = torch.stack(samples).to(self.dummy)
        labels = torch.tensor(labels, dtype=torch.int64).to(self.dummy.device)
        return samples, labels
class FourierEncoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(1, self.half_dim))

    def forward(self, t: Float[Array,'bs 1 1 1']) -> Float[Array,'bs dim']:
        t = t.view(-1, 1) # (bs, 1)
        freqs = t * self.weights * 2 * math.pi # (bs, half_dim)
        sin_embed = torch.sin(freqs) # (bs, half_dim)
        cos_embed = torch.cos(freqs) # (bs, half_dim)
        return torch.cat([sin_embed, cos_embed], dim=-1) * math.sqrt(2) # (bs, dim
class Matcher(nn.Module):
    def __init__(self, d_in:Int,latent_model: FeatureEncoder,out_channels:int,img_size:int=32 ):
        
        super().__init__()
        self.t_embeder= FourierEncoder(dim=d_in)
        self.y_embedder = Embedding(num_embeddings=11, embedding_dim=d_in)
        
        self.init_conv = nn.ConvTranspose2d(
    in_channels=d_in, out_channels=64, kernel_size=4, stride=2
)
        self.docoder=nn.Sequential(
            nn.Upsample(size=(img_size, img_size), mode='bilinear', align_corners=False),
            nn.GELU(), 
            Conv2d(64, 8, kernel_size=1, stride=1), 
            nn.BatchNorm2d(8) ,
         
                
            Conv2d(8, out_channels, kernel_size=1, stride=1),
            #nn.GELU(),
            #nn.BatchNorm2d(out_channels) 
        )
        
        self.latent_model = latent_model
        
        
        
    def forward(self, x: Float[Array,"bs 1 32 32"], t: Float[Array,"bs 1 1 1"], y: Float[Array,"bs ..."]) -> Float[Array,"bs 1 32 32"]:
        t_embed = self.t_embeder(t)
        y_embed = self.y_embedder(y)
        latent = self.latent_model(x)
        x_y=einsum(latent,y_embed,'b seq d ,b d-> b seq d')
        x_t=einsum(x_y, t_embed, 'b seq d , b d -> b seq d ')
        
        x_t = rearrange(x_t, 'b seq d -> b d seq 1')
        x_t = self.init_conv(x_t)
       
        x_t = self.docoder(x_t)
    
        return x_t

class VecField(nn.Module):
    def __init__(self,matcher:Matcher):
        super().__init__()
        self.matcher=matcher
        
    def forward(self, x: Float[Array,"bs c h w"], t: Float[Array,"bs 1 1 1"], y: Float[Array,"bs ..."]) -> Float[Array, "bs c h w"]:
        x=self.matcher(x,t,y)
        return x


class CFG(nn.Module):
    def __init__(self,path: GaussianConditionalProbabilityPath, model: VecField, eta: float,):
      super().__init__()
      self.eta=eta
      self.path=path
      self.model=model
    
    def forward( self,batch_size: int)->Float[Array,'...']:
        z, y = self.path.p_data.sample(batch_size)
        xi = torch.rand(y.shape[0]).to(y.device)
        y[xi < self.eta] = 10.0
        
        t = torch.rand(batch_size,1,1,1).to(z) # (bs, 1, 1, 1)
        x = self.path.sample_conditional_path(z,t) # (bs, 1, 32, 32)
        
        
        ut_theta = self.model(x,t,y) # (bs, 1, 32, 32)
        ut_ref = self.path.conditional_vector_field(x,z,t) # (bs, 1, 32, 32)
        error = torch.einsum('bchw -> b', torch.square(ut_theta - ut_ref)) # (bs,)
        return torch.mean(error)
    

