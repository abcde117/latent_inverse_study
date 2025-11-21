from flow_model import CFG
from einops import rearrange, einsum
import einx
import math
from jaxtyping import Float, Bool, Int,Array
from typing import *
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from flow_model import *
from flow_utils import FlowLogger
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm


class LatentCFG(nn.Module):
    def __init__(self,forward_model,inverse_model,path: GaussianConditionalProbabilityPath, eta: float,):
      super().__init__()
      self.eta=eta
      self.path=path
      self.forward_model=forward_model
      self.inverse_model=inverse_model
    
    def forward( self,batch_size: int)->Float[Array,'...']:
        z, y = self.path.p_data.sample(batch_size)
        xi = torch.rand(y.shape[0]).to(y.device)
        y[xi < self.eta] = 10.0
        
        t = torch.rand(batch_size,1,1,1).to(z) # (bs, 1, 1, 1)
        x = self.path.sample_conditional_path(z,t) # (bs, 1, 32, 32)
        
        
        ut_theta_latent = self.forward_model(x,t,y) # (bs, seq,d)
        ut_ref = self.path.conditional_vector_field(x,z,t) # (bs, 1, 32, 32)
        ut_ref_latent=self.inverse_model(ut_ref)# (bs, seq,d)
        error = einsum(torch.square( ut_theta_latent-ut_ref_latent),'b seq d -> b')
        return torch.mean(error)

''''
class InverseCFG(nn.Module):
    def __init__(self,decoder_model,inverse_model,path: GaussianConditionalProbabilityPath, eta: float,):
      super().__init__()
      self.eta=eta
      self.path=path
      self.decoder_model=decoder_model
      self.inverse_model=inverse_model
    
    def forward( self,batch_size: int)->Float[Array,'...']:
        z, y = self.path.p_data.sample(batch_size)
        xi = torch.rand(y.shape[0]).to(y.device)
        y[xi < self.eta] = 10.0
        
        t = torch.rand(batch_size,1,1,1).to(z) # (bs, 1, 1, 1)
        x = self.path.sample_conditional_path(z,t) # (bs, 1, 32, 32)
        
        ut_ref = self.path.conditional_vector_field(x,z,t) # (bs, 1, 32, 32)
        latent=self.inverse_model(ut_ref)
        ut_decode=self.decoder_model(latent)
        error = einsum(torch.square( ut_decode- ut_ref),'b c h w -> b')
        return torch.mean(error)'''

class InverseutCFG(nn.Module):
    def __init__(self,decoder_model,inverse_model,path: GaussianConditionalProbabilityPath, eta: float,):
      super().__init__()
      self.eta=eta
      self.path=path
      self.decoder_model=decoder_model
      self.inverse_model=inverse_model
    
    def forward( self,batch_size: int)->Float[Array,'...']:
        z, y = self.path.p_data.sample(batch_size)
        xi = torch.rand(y.shape[0]).to(y.device)
        y[xi < self.eta] = 10.0
        
        t = torch.rand(batch_size,1,1,1).to(z) # (bs, 1, 1, 1)
        x = self.path.sample_conditional_path(z,t) # (bs, 1, 32, 32)
        
        ut_ref = self.path.conditional_vector_field(x,z,t) # (bs, 1, 32, 32)
        latent=self.inverse_model(ut_ref)
        ut_decode=self.decoder_model(latent,t=t,y=y)
        error = einsum(torch.square( ut_decode- ut_ref),'b c h w -> b')
        return torch.mean(error)
    
class contexter(nn.Module):
     def __init__(self,latent_model,d_in:Int):
        super().__init__()
        self.t_embeder= FourierEncoder(dim=d_in)
        self.y_embedder = Embedding(num_embeddings=11, embedding_dim=d_in)
        self.latent_model=latent_model
    
     def forward(self, x: Float[Array,"bs 1 32 32"], t: Float[Array,"bs 1 1 1"], y: Float[Array,"bs ..."]) -> Float[Array,"b seq d"]:
        t_embed = self.t_embeder(t)
        y_embed = self.y_embedder(y)
        latent_output = self.latent_model(x)
        x_y=einsum(latent_output,y_embed,'b seq d ,b d-> b seq d')
        x_t=einsum(x_y, t_embed, 'b seq d , b d -> b seq d ')
        return x_t
class LatentVecField(nn.Module):
    def __init__(self,latent,d_in:Int):
        super().__init__()
        self.contexter=contexter(latent,d_in)
        
    def forward(self, x: Float[Array,"bs c h w"], t: Float[Array,"bs 1 1 1"], y: Float[Array,"bs ..."]) -> Float[Array, "bs seq  d"]:
        x=self.contexter(x,t,y)
        return x
    
class vecfield(nn.Module):
    def __init__(self,encoder,decoder):
        super().__init__()
        self.encoder=encoder
        self.decoder=decoder
    
    
    def forward(self, x: Float[Array,"bs 1 32 32"], t: Float[Array,"bs 1 1 1"], y: Float[Array,"bs ..."]) -> Float[Array,"bs 1 32 32"]:
        latent=self.encoder(x,t=t,y=y)
        return self.decoder(latent,t=t,y=y)

class Utdecoder(nn.Module):
    def __init__ (self, d_in: int, d_out: int, img_size=32):
        super().__init__()
        self.init_conv = nn.ConvTranspose2d(in_channels=d_in, out_channels=64, kernel_size=4, stride=2)
        self.t_embeder= FourierEncoder(dim=d_in)
        self.y_embedder = Embedding(num_embeddings=11, embedding_dim=d_in)
        self.decoder = nn.Sequential(
            nn.Upsample(size=(img_size, img_size), mode='bilinear', align_corners=False),
            nn.GELU(),
            Conv2d(64, 8, kernel_size=1, stride=1),
            nn.BatchNorm2d(8),
            Conv2d(8, out_channels=d_out, kernel_size=1, stride=1),
            #nn.GELU(),
            #nn.BatchNorm2d(d_out)
        )

    def forward(self, x: Float[Array,'b seq d'], t: Float[Array,"bs 1 1 1"], y: Float[Array,"bs ..."]) -> Float[Array,"bs c h w"]:
        t_embed = self.t_embeder(t)
        y_embed = self.y_embedder(y)
        x_y=einsum(x,y_embed,'b seq d ,b d-> b seq d')
        x_t=einsum(x_y, t_embed, 'b seq d , b d -> b seq d ')
        x = rearrange(x, 'b seq d -> b d seq 1')
        x = self.init_conv(x)
        x = self.decoder(x)
        return x

class LatentFlowModelTrainer2:
 def __init__(self,device, forward_model: nn.Module,inverse_model: nn.Module,loss_:LatentCFG,logger=FlowLogger,):
        super().__init__()
        self.forward_model = forward_model
        self.inverse_model=inverse_model
        self.get_loss=loss_
        self.logger = logger
 def get_optimizer(self, lr: float):
        return torch.optim.Adam(self.forward_model.parameters(), lr=lr)
 
 def train(self,  num_epochs: int, batch_size: int, device, lr: float = 1e-3) ->Float[Array, ""]:
        self.forward_model.to(device)
        self.inverse_model.to(device)
        forward_opt = self.get_optimizer(lr)
        self.forward_model.train()
        self.inverse_model.eval()
        
        pbar = tqdm(enumerate(range(num_epochs)))
        for idx, epoch in pbar:
            forward_opt.zero_grad()
           
            loss = self.get_loss(batch_size)
            loss.backward()
            forward_opt.step()
          
            
            pbar.set_description(f'Epoch {idx}, loss: {loss.item():.3f}')
            self.forward_model.eval()
            self.inverse_model.eval()
            self.get_loss.eval()
            with torch.no_grad():
                val_loss = self.get_loss(batch_size)

            if self.logger:
                self.logger.log(epoch,loss.item(), val_loss.item())

            pbar.set_postfix({
                "train": f"{loss.item():.4f}",
                "val": f"{val_loss.item():.4f}"
            })
            
class LatentFlowModelTrainer:
 def __init__(self,device, forward_model: nn.Module,inverse_model: nn.Module,loss_:LatentCFG,logger=FlowLogger,):
        super().__init__()
        self.forward_model = forward_model
        self.inverse_model=inverse_model
        self.get_loss=loss_
        self.logger = logger
        
        
        
 def get_optimizer(self, lr: float):
        return torch.optim.Adam(self.forward_model.parameters(), lr=lr),torch.optim.Adam(self.inverse_model.parameters(), lr=lr)
 
 def train(self,  num_epochs: int, batch_size: int, device, lr: float = 1e-3) ->Float[Array, ""]:
        self.forward_model.to(device)
        self.inverse_model.to(device)
        forward_opt,inverse_opt = self.get_optimizer(lr)
        self.forward_model.train()
        self.inverse_model.train()
        
        pbar = tqdm(enumerate(range(num_epochs)))
        for idx, epoch in pbar:
            forward_opt.zero_grad()
            inverse_opt.zero_grad()
            loss = self.get_loss(batch_size)
            loss.backward()
            forward_opt.step()
            inverse_opt.step()
            
            pbar.set_description(f'Epoch {idx}, loss: {loss.item():.3f}')
            self.forward_model.eval()
            self.inverse_model.eval()
            self.get_loss.eval()
            with torch.no_grad():
                val_loss = self.get_loss(batch_size)

            if self.logger:
                self.logger.log(epoch,loss.item(), val_loss.item())

            pbar.set_postfix({
                "train": f"{loss.item():.4f}",
                "val": f"{val_loss.item():.4f}"
            })
            
            
            

            
class LatentFlowExperimentRunner:
    def __init__(self,
                 models: dict,  # e.g. {"ViT": {"forward": ..., "inverse": ...}, ...}
                 cfg_class: LatentCFG,
                 path,
                 num_epochs: int,
                 batch_size: int,
                 device,
                 eta: float = 0.1,
                 lr: float = 1e-3):
        self.models = models
        self.cfg_class = cfg_class
        
        
        self.path = path
        self.eta = eta
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device

    def run(self):
        for name, model_pair in self.models.items():
            print(f"\n>>> Training Model: {name}")
            forward_model = model_pair["forward"]
            inverse_model = model_pair["inverse"]

            logger = FlowLogger(model_name=name)

           
            cfg_loss = self.cfg_class(
                forward_model=forward_model,
                inverse_model=inverse_model,
                path=self.path,
                eta=self.eta
            )

          
            trainer = LatentFlowModelTrainer(
                device=self.device,
                forward_model=forward_model,
                inverse_model=inverse_model,
                loss_=cfg_loss,
                logger=logger
            )

        
            trainer.train(
                num_epochs=self.num_epochs,
                batch_size=self.batch_size,
                lr=self.lr,
                device=self.device,
            )

            logger.save(f"{name}_flow_loss.csv")
            

            
class LatentFlowExperimentRunner2:
    def __init__(self,
                 models: dict,  # e.g. {"ViT": {"forward": ..., "inverse": ...}, ...}
                 cfg_class: LatentCFG,
                 path,
                 num_epochs: int,
                 batch_size: int,
                 device,
                 eta: float = 0.1,
                 lr: float = 1e-3):
        self.models = models
        self.cfg_class = cfg_class
        self.path = path
        self.eta = eta
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device

    def run(self):
        for name, model_pair in self.models.items():
            print(f"\n>>> Training Model: {name}")
            forward_model = model_pair["forward"]
            inverse_model = model_pair["inverse"]

            logger = FlowLogger(model_name=name)

           
            cfg_loss = self.cfg_class(
                forward_model=forward_model,
                inverse_model=inverse_model,
                path=self.path,
                eta=self.eta
            )

          
            trainer = LatentFlowModelTrainer2(
                device=self.device,
                forward_model=forward_model,
                inverse_model=inverse_model,
                loss_=cfg_loss,
                logger=logger
            )

        
            trainer.train(
                num_epochs=self.num_epochs,
                batch_size=self.batch_size,
                lr=self.lr,
                 device=self.device,
            )

            logger.save(f"{name}_flow_loss.csv")



class InverseFlowModelTrainer:
 def __init__(self,device, decoder_model: nn.Module,inverse_model: nn.Module,loss_:InverseutCFG,logger=FlowLogger,):
        super().__init__()
        self.decoder_model = decoder_model
        self.inverse_model=inverse_model
        self.get_loss=loss_
        self.logger = logger
 def get_optimizer(self, lr: float):
        return torch.optim.Adam(self.decoder_model.parameters(), lr=lr)
 
 def train(self,  num_epochs: int, batch_size: int, device, lr: float = 1e-3) ->Float[Array, ""]:
        self.decoder_model.to(device)
        self.inverse_model.to(device)
        opt = self.get_optimizer(lr)
        self.decoder_model.train()
        self.inverse_model.eval()
        
        pbar = tqdm(enumerate(range(num_epochs)))
        for idx, epoch in pbar:
            opt.zero_grad()
            loss = self.get_loss(batch_size)
            loss.backward()
            opt.step()
        
            pbar.set_description(f'Epoch {idx}, loss: {loss.item():.3f}')
            self.decoder_model.eval()
            self.inverse_model.eval()
            self.get_loss.eval()
            with torch.no_grad():
                val_loss = self.get_loss(batch_size)

            if self.logger:
                self.logger.log(epoch,loss.item(), val_loss.item())

            pbar.set_postfix({
                "train": f"{loss.item():.4f}",
                "val": f"{val_loss.item():.4f}"
            })
            
            
            

class InverseFlowModelTrainer2:
 def __init__(self,device, decoder_model: nn.Module,inverse_model: nn.Module,loss_:InverseutCFG,logger=FlowLogger,):
        super().__init__()
        self.decoder_model = decoder_model
        self.inverse_model=inverse_model
        self.get_loss=loss_
        self.logger = logger
 def get_optimizer(self, lr: float):
        return torch.optim.Adam(self.decoder_model.parameters(), lr=lr)
 
 def train(self,  num_epochs: int, batch_size: int, device, lr: float = 1e-3) ->Float[Array, ""]:
        self.decoder_model.to(device)
        self.inverse_model.to(device)
        opt = self.get_optimizer(lr)
        self.decoder_model.train()
        self.inverse_model.eval()
        
        pbar = tqdm(enumerate(range(num_epochs)))
        for idx, epoch in pbar:
            opt.zero_grad()
            loss = self.get_loss(batch_size)
            loss.backward()
            opt.step()
        
            pbar.set_description(f'Epoch {idx}, loss: {loss.item():.3f}')
            self.decoder_model.eval()
            self.inverse_model.eval()
            self.get_loss.eval()
            with torch.no_grad():
                val_loss = self.get_loss(batch_size)

            if self.logger:
                self.logger.log(epoch,loss.item(), val_loss.item())

            pbar.set_postfix({
                "train": f"{loss.item():.4f}",
                "val": f"{val_loss.item():.4f}"
            })


class InverseFlowExperimentRunner:
    def __init__(self,
                 models: dict,  
                 cfg_class: InverseutCFG,
                 path,
                 num_epochs: int,
                 batch_size: int,
                 device,
                 eta: float = 0.1,
                 lr: float = 1e-3):
        self.models = models
        self.cfg_class = cfg_class
        self.path = path
        self.eta = eta
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device

    def run(self):
        for name, model_pair in self.models.items():
            print(f"\n>>> Training Model: {name}")
            decoder_model = model_pair["decoder"]
            inverse_model = model_pair["inverse"]

            logger = FlowLogger(model_name=name)

           
            cfg_loss = self.cfg_class(
                decoder_model=decoder_model,
                inverse_model=inverse_model,
                path=self.path,
                eta=self.eta
            )

          
            trainer = InverseFlowModelTrainer(
                device=self.device,
                decoder_model=decoder_model,
                inverse_model=inverse_model,
                loss_=cfg_loss,
                logger=logger
            )

        
            trainer.train(
                num_epochs=self.num_epochs,
                batch_size=self.batch_size,
                lr=self.lr,
                device=self.device
            )

            logger.save(f"{name}_flow_loss.csv")


class InverseFlowExperimentRunner2:
    def __init__(self,
                 models: dict,  
                 cfg_class: InverseutCFG,
                 path,
                 num_epochs: int,
                 batch_size: int,
                 device,
                 eta: float = 0.1,
                 lr: float = 1e-3):
        self.models = models
        self.cfg_class = cfg_class
        self.path = path
        self.eta = eta
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device

    def run(self):
        for name, model_pair in self.models.items():
            print(f"\n>>> Training Model: {name}")
            decoder_model = model_pair["decoder"]
            inverse_model = model_pair["inverse"]

            logger = FlowLogger(model_name=name)

           
            cfg_loss = self.cfg_class(
                decoder_model=decoder_model,
                inverse_model=inverse_model,
                path=self.path,
                eta=self.eta
            )

          
            trainer = InverseFlowModelTrainer2(
                device=self.device,
                decoder_model=decoder_model,
                inverse_model=inverse_model,
                loss_=cfg_loss,
                logger=logger
            )

        
            trainer.train(
                num_epochs=self.num_epochs,
                batch_size=self.batch_size,
                lr=self.lr,
                 device=self.device,
            )

            logger.save(f"{name}_flow_loss.csv")

class biencoder(nn.Module):
    def __init__(self, encoder, invencoder):
        super().__init__()
        self.encoder = encoder
        self.invencoder = invencoder

    def forward(self, x: Float[Array, "bs c h w"], t: Float[Array, "bs 1 1 1"], y: Float[Array, "bs ..."]) -> Float[Array, "bs c h w"]:
        x = self.encoder(x, t, y)        
        x = self.invencoder(x)           
        return x
    
class FullCFG(nn.Module):
    def __init__(self,forward_model,inverse_model,decoder_model,path: GaussianConditionalProbabilityPath, eta: float,):
      super().__init__()
      self.eta=eta
      self.path=path
      self.forward_model=forward_model
      self.inverse_model=inverse_model
      self.decoder_model=decoder_model
    
    def forward( self,batch_size: int)->Float[Array,'...']:
        z, y = self.path.p_data.sample(batch_size)
        xi = torch.rand(y.shape[0]).to(y.device)
        y[xi < self.eta] = 10.0
        
        t = torch.rand(batch_size,1,1,1).to(z) 
        x = self.path.sample_conditional_path(z,t) 
        
        
        ut_theta_latent = self.forward_model(x,t,y) 
        ut_ref = self.path.conditional_vector_field(x,z,t)
        ut_ref_latent=self.inverse_model(ut_ref)
        ut_inverse_path=self.decoder_model(ut_ref_latent,t=t,y=y)
        ut_forward_path=self.decoder_model(ut_theta_latent,t=t,y=y)
        loss_latent=einsum(torch.square(ut_theta_latent-ut_ref_latent),'b seq d ->b').mean()
        loss_inverse_path=einsum(torch.square(ut_inverse_path- ut_ref),'b c h w ->b').mean()
        loss_forward_path=einsum(torch.square(ut_forward_path- ut_ref),'b c h w ->b').mean()
        loss_diff_path= einsum(torch.square(ut_forward_path-ut_inverse_path),'b c h w ->b').mean()
        #error = einsum(torch.square( ut_theta_latent-ut_ref_latent),'b seq d -> b')
        return ((loss_inverse_path+ loss_forward_path)/2+loss_latent+loss_diff_path)/3
    
    
class FullFlowModelTrainer:
 def __init__(self,device, forward_model: nn.Module,inverse_model: nn.Module,decoder_model:nn.Module,loss_:FullCFG,logger=FlowLogger,):
        super().__init__()
        self.forward_model = forward_model
        self.inverse_model=inverse_model
        self.decoder_model=decoder_model
        self.get_loss=loss_
        self.logger = logger
 def get_optimizer(self, lr: float):
        return (torch.optim.Adam(self.forward_model.parameters(), lr=lr),
    torch.optim.Adam(self.inverse_model.parameters(), lr=lr),
    torch.optim.Adam(self.inverse_model.parameters(), lr=lr))
                                
 
 def train(self,  num_epochs: int, batch_size: int, device, lr: float = 1e-3) ->Float[Array, ""]:
        self.forward_model.to(device)
        self.inverse_model.to(device)
        self.decoder_model.to(device)
        forward_opt,inverse_opt,decoder_opt = self.get_optimizer(lr)
        self.forward_model.train()
        self.inverse_model.train()
        self.decoder_model.train()
        
        pbar = tqdm(enumerate(range(num_epochs)))
        for idx, epoch in pbar:
            forward_opt.zero_grad()
            inverse_opt.zero_grad()
            decoder_opt.zero_grad()
            loss = self.get_loss(batch_size)
            loss.backward()
            forward_opt.step()
            inverse_opt.step()
            decoder_opt.step()
            
            pbar.set_description(f'Epoch {idx}, loss: {loss.item():.3f}')
            self.forward_model.eval()
            self.inverse_model.eval()
            self.decoder_model.eval()
            self.get_loss.eval()
            with torch.no_grad():
                val_loss = self.get_loss(batch_size)

            if self.logger:
                self.logger.log(epoch,loss.item(), val_loss.item())

            pbar.set_postfix({
                "train": f"{loss.item():.4f}",
                "val": f"{val_loss.item():.4f}"
            })
            
            
class FullFlowExperimentRunner:
    def __init__(self,
                 models: dict,  # e.g. {"ViT": {"forward": ..., "inverse": ...}, ...}
                 cfg_class: FullCFG,
                 path,
                 num_epochs: int,
                 batch_size: int,
                 device,
                 eta: float = 0.1,
                 lr: float = 1e-3):
        self.models = models
        self.cfg_class = cfg_class
        self.path = path
        self.eta = eta
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device

    def run(self):
        for name, model_pair in self.models.items():
            print(f"\n>>> Training Model: {name}")
            forward_model = model_pair["forward"]
            inverse_model = model_pair["inverse"]
            decoder_model= model_pair["decoder"]

            logger = FlowLogger(model_name=name)

           
            cfg_loss = self.cfg_class(
                forward_model=forward_model,
                inverse_model=inverse_model,
                decoder_model=decoder_model,
                path=self.path,
                eta=self.eta
            )

          
            trainer = FullFlowModelTrainer(
                device=self.device,
                forward_model=forward_model,
                inverse_model=inverse_model,
                decoder_model=decoder_model,
                loss_=cfg_loss,
                logger=logger
            )

        
            trainer.train(
                num_epochs=self.num_epochs,
                batch_size=self.batch_size,
                lr=self.lr,
                 device=self.device,
            )

            logger.save(f"{name}_flow_loss.csv")