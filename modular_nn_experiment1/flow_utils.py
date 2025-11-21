from einops import rearrange, einsum
import einx
import math
from jaxtyping import Float, Bool, Int,Array
from typing import *
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm
from models_ import FeatureEncoder
from flow_model import *

class FlowLogger:
    def __init__(self, model_name):
        self.model_name = model_name
        self.loss_records = []

    def log(self, epoch, train_loss, val_loss=None):
        self.loss_records.append({
            "Model": self.model_name,
            "Epoch": epoch,
            "TrainLoss": train_loss,
            "ValLoss": val_loss if val_loss is not None else -1
        })

    def save(self, path="loss_log.csv"):
        import pandas as pd
        df = pd.DataFrame(self.loss_records)
        df.to_csv(path, index=False)
        
class FlowModelTrainer:
 def __init__(self,device, model: nn.Module,loss_:CFG,logger=FlowLogger,):
        super().__init__()
        self.model = model
        self.get_loss=loss_
        self.logger = logger
 def get_optimizer(self, lr: float):
        return torch.optim.Adam(self.model.parameters(), lr=lr)
 
 def train(self,  num_epochs: int, batch_size: int, device, lr: float = 1e-3) ->Float[Array, ""]:
        self.model.to(device)
        opt = self.get_optimizer(lr)
        self.model.train()
        
        pbar = tqdm(enumerate(range(num_epochs)))
        for idx, epoch in pbar:
            opt.zero_grad()
            loss = self.get_loss(batch_size)
            loss.backward()
            opt.step()
            pbar.set_description(f'Epoch {idx}, loss: {loss.item():.3f}')
            self.model.eval()
            self.get_loss.eval()
            with torch.no_grad():
                val_loss = self.get_loss(batch_size)

            if self.logger:
                self.logger.log(epoch,loss.item(), val_loss.item())

            pbar.set_postfix({
                "train": f"{loss.item():.4f}",
                "val": f"{val_loss.item():.4f}"
            })
        
class FlowExperimentRunner:
    def __init__(self, models, cfg_class, path, num_epochs: int, batch_size: int,
                 device, eta: float = 0.1, lr: float = 1e-3):
        self.models = models
        self.cfg_class = cfg_class
        self.path = path
        self.eta = eta
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device

    def run(self):
        for name, model in self.models.items():
            print(f"\n>>> Training Model: {name}")
            logger = FlowLogger(model_name=name)

            cfg_loss = self.cfg_class(path=self.path, model=model, eta=self.eta)
            trainer = FlowModelTrainer(model=model, loss_=cfg_loss,
                                       device=self.device, logger=logger)

            trainer.train(num_epochs=self.num_epochs,
                          batch_size=self.batch_size,
                          lr=self.lr,
                          device=self.device)

            logger.save(f"{name}_flow_loss.csv")