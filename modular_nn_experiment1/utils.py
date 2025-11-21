import torch
import torch.nn as nn
import numpy as np
import random     
from typing import Union
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

import copy
from tqdm import tqdm
from models_ import *

from torchvision import datasets, transforms


# utils.py  


#
MiB = 1024 ** 2

def model_size_b(model: nn.Module) -> int:
    size = 0
    for param in model.parameters():
        size += param.nelement() * param.element_size()
    for buf in model.buffers():
        size += buf.nelement() * buf.element_size()
    return size

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_model_params(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# datalogger
class ExperimentLogger:
    def __init__(self, model_name):
        self.model_name = model_name
        self.summary_records = []
        self.detail_records = []

    def log_summary(self, fold, epoch, train_acc, val_acc):
        self.summary_records.append({
            "Model": self.model_name,
            "Fold": fold,
            "Epoch": epoch,
            "TrainAcc": train_acc,
            "ValAcc": val_acc
        })

    def log_predictions(self, preds_batch):
        self.detail_records.extend(preds_batch)

    def save(self, summary_path, detail_path):
        pd.DataFrame(self.summary_records).to_csv(summary_path, index=False)
        pd.DataFrame(self.detail_records).to_csv(detail_path, index=False)





#trainer
class ModelTrainer:
    def __init__(self, model, classifier, criterion, optimizer, device):
        self.model = model
        self.classifier = classifier
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_epoch(self, loader):
        self.model.train()
        self.classifier.train()
        total, correct = 0, 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            feats = self.model(x)
            logits = self.classifier(feats)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        return correct / total

    def eval_and_collect(self, loader, model_name, fold, epoch):
        self.model.eval()
        self.classifier.eval()
        total, correct = 0, 0
        batch_records = []
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                feats = self.model(x)
                logits = self.classifier(feats)
                preds = logits.argmax(dim=1)
                total += y.size(0)
                correct += (preds == y).sum().item()
                for i in range(len(x)):
                    batch_records.append({
                        "Model": model_name,
                        "Fold": fold,
                        "Epoch": epoch,
                        "true_label": y[i].item(),
                        "pred_label": preds[i].item(),
                        "correct": int(preds[i] == y[i])
                    })
        acc = correct / total
        return acc, batch_records

# ExperimentRunner

class ExperimentRunner:
    def __init__(self, models, dataset, num_folds, num_epochs, batch_size, lr, classfier_in):
        self.models = models
        self.dataset = dataset
        self.num_folds = num_folds
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.classfier_in = classfier_in
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.CrossEntropyLoss()
        set_seed(42)

    def run(self):
        kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)

        for name, model_instance in self.models.items():
            print(f"\\n>>> Model: {name}", count_model_params(model_instance))
            logger = ExperimentLogger(model_name=name)

            for fold, (train_idx, val_idx) in enumerate(kf.split(self.dataset)):
                train_loader = DataLoader(Subset(self.dataset, train_idx), batch_size=self.batch_size, shuffle=True)
                val_loader = DataLoader(Subset(self.dataset, val_idx), batch_size=self.batch_size, shuffle=False)

                model = copy.deepcopy(model_instance).to(self.device)
                classifier = UniversalClassifier(self.classfier_in, d_out=10).to(self.device) # Added d_out=10
                optimizer = torch.optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=self.lr)

                if isinstance(model, PCAMLP):
                    model.fit_pca(Subset(self.dataset, train_idx), self.device)

                trainer = ModelTrainer(model, classifier, self.criterion, optimizer, self.device)

                for epoch in range(1, self.num_epochs + 1):
                    train_acc = trainer.train_epoch(train_loader)
                    val_acc, preds = trainer.eval_and_collect(val_loader, name, fold, epoch)
                    logger.log_summary(fold, epoch, train_acc, val_acc)
                    logger.log_predictions(preds)
                    print(f"{name} Fold {fold} Epoch {epoch}: Train={train_acc:.4f}, Val={val_acc:.4f}")

            logger.save(f"{name}_summary.csv", f"{name}_details.csv")