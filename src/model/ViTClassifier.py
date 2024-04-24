import lightning as L
import torch

class LitViTClassifier(L.LightningModule):
    def __init__(self, model, criterion, lr) -> None:
        super().__init__()
        self.model =  model
        self.criterion = criterion
        self.loss = criterion
        self.lr = lr
        
    def forward(self, inputs):
        return self.model(inputs)
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)
    
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
    