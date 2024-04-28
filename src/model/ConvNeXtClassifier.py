import lightning as L
import torch
from dataclasses import dataclass
import pytorch_warmup as warmup
from transformers import get_cosine_schedule_with_warmup


class LitConvNeXtClassifier(L.LightningModule):
    def __init__(self, model=None, criterion=None, lr=None) -> None:
        super().__init__()
        self.model =  model
        self.criterion = criterion
        self.lr = lr
        
    def forward(self, inputs):
        return self.model(inputs)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.lr, betas=[0.9, 0.999], 
            weight_decay=0.05,
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=3000,
            num_training_steps=2e5,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
            }
        }
    
    def training_step(self, batch, batch_idx):
        x,y = batch.pixel_values, batch.labels
        out = self(x)
        loss = self.criterion(out.logits, y)
        self.log("train_loss", loss, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x,y = batch.pixel_values, batch.labels
        out = self(x)
        loss = self.criterion(out.logits, y)
        
        # calculate acc
        labels_hat = torch.argmax(out.logits, dim=1)
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)

        self.log_dict({'val_loss': loss, 'val_accuracy': val_acc}, on_epoch=True)
        
    def predict_step(self, batch, batch_idx):
        x =  batch.pixel_values
        out = self(x)
        labels_hat = torch.argmax(out.logits, dim=1)
        return labels_hat
    
        