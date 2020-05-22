import math

from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from tensorboardX import SummaryWriter

from .criterion import DefaultCriterion


class Trainer(object):
    def __init__(self, model, train_dataloader, valid_dataloader, **kwargs):
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        self.device = kwargs.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        if kwargs.get("loss_fn", None) is not None:
            self._loss_fn = kwargs.get("loss_fn")
        else:
            self._loss_fn = DefaultCriterion(kwargs.get("criterion"))

        self.metrics = kwargs.get("metrics", [])

        self.lr = kwargs.get("lr", 0.001)
        self.optimizer = kwargs.get("optimizer", torch.optim.Adam(self.model.parameters(), lr=self.lr))
        self.epochs = kwargs.get("epochs", 100)

        self.checkpoint = kwargs.get("checkpoint", 0)
        self.valid_interval = kwargs.get("valid_interval", 1)
        self.save_dir = Path(kwargs.get("save_dir", "exp"))

        if kwargs.get("log_dir", None) is not None:
            self.writer = SummaryWriter(kwargs.get("log_dir"))
        else:
            self.writer = SummaryWriter()

        self.reset()

    @property
    def log_dir(self):
        return self.writer.logdir

    def reset(self):
        self.steps = 0
        self.best_val_loss = float("inf")
        self.start_epoch = 0

        self.save_dir.mkdir(parents=True, exist_ok=True)

    def train(self):
        self.model.to(self.device)

        try:
            self._loss_fn.to(self.device)
        except:
            pass

        for epoch in range(self.start_epoch, self.epochs):
            self.model.train()
            train_loss, train_scores = self._run_one_epoch(epoch)

            self.writer.add_scalar("avg_loss", train_loss, epoch)

            for name, score in train_scores.items():
                self.writer.add_scalar(name, score, epoch)

            # Save model if (epoch + 1) % checkpoint == 0
            if self.checkpoint != 0:
                if (epoch + 1) % self.checkpoint == 0:
                    output_path = self.save_dir / f"{epoch + 1}.pkl"
                    torch.save(self.model.state_dict(), str(output_path))

            # Validation
            if (epoch + 1) % self.valid_interval == 0:
                self.model.eval()
                val_loss, val_scores = self._run_one_epoch(epoch, valid=True)

                self.writer.add_scalar("val_loss", val_loss, epoch)

                for name, score in val_scores.items():
                    self.writer.add_scalar(f"val_{name}", score, epoch)

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    output_path = self.save_dir / f"best.pkl"
                    torch.save(self.model.state_dict(), str(output_path))

    def _run_one_epoch(self, epoch, valid=False):
        total_loss = 0
        total_scores = defaultdict(int)
        dataloader = self.train_dataloader if not valid else self.valid_dataloader

        with tqdm(enumerate(dataloader), total=math.ceil(len(dataloader.dataset) / dataloader.batch_size), unit="iter") as t:
            for i, (data) in t:
                t.set_description(f"Epoch {epoch + 1}") if not valid else t.set_description("Validation")

                loss, pred, y = self._loss_fn(self.model, data)

                for metric in self.metrics:
                    score = metric(pred, y)
                    total_scores[metric.name] += score

                if not valid:
                    self.writer.add_scalar("loss", loss.item(), self.steps)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    self.steps += 1

                total_loss += loss.item()
                avg_loss = total_loss / (i + 1)

                t.set_postfix(steps=self.steps, avg_loss=avg_loss, loss=loss.item())
        
        avg_scores = {
            name: score / i
            for name, score in total_scores.items()
        }

        return avg_loss, avg_scores

