import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassConfusionMatrix
from transformers import AutoProcessor, AutoModel
from PIL import Image
import os
import pandas as pd
from sklearn.model_selection import train_test_split


class MVSAImageDataset(Dataset):

    def __init__(self, data, processor):
        self.data = data
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, soft_label = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)
        return pixel_values, torch.tensor(soft_label, dtype=torch.float32)

def single_collate_fn(batch):
    pixel_values, labels = zip(*batch)
    return torch.stack(pixel_values), torch.stack(labels)


class VisionLanguageModel(pl.LightningModule):
    


    def __init__(self, model_name="openai/clip-vit-base-patch32", num_classes=3, lr_head=1e-3, lr_backbone=1e-6, freeze_backbone=True):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = AutoModel.from_pretrained(model_name)
        
        if hasattr(self.backbone.config, "projection_dim"):
            embed_dim = self.backbone.config.projection_dim

        elif hasattr(self.backbone.config, "hidden_size"):
            embed_dim = self.backbone.config.hidden_size
        
        else:
            embed_dim = self.backbone.config.vision_config.hidden_size

        self.classifier = nn.Linear(embed_dim, num_classes)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
       
        self.criterion = nn.KLDivLoss(reduction='batchmean')
        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average="weighted")
        self.test_acc = MulticlassAccuracy(num_classes=num_classes)
        self.test_f1 = MulticlassF1Score(num_classes=num_classes, average="weighted")
        self.conf_mat = MulticlassConfusionMatrix(num_classes=num_classes)


    def forward(self, pixel_values):
        
        if hasattr(self.backbone, "get_image_features"):
            feats = self.backbone.get_image_features(pixel_values=pixel_values)

        elif hasattr(self.backbone, "vision_model"):
            feats = self.backbone.vision_model(pixel_values=pixel_values).pooler_output
        
        else:
            outputs = self.backbone(pixel_values=pixel_values)            
            feats = getattr(outputs, "image_embeds", outputs.pooler_output)
        
        feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
        logits = self.classifier(feats)
        return logits

    def training_step(self, batch, batch_idx):
        pixel_values, labels = batch
        logits = self(pixel_values)
        log_probs = F.log_softmax(logits, dim=1)
        loss = self.criterion(log_probs, labels)

        preds = torch.argmax(logits, dim=-1)
        target = torch.argmax(labels, dim=-1)
        
        self.train_acc(preds, target)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pixel_values, labels = batch
        logits = self(pixel_values)
        log_probs = F.log_softmax(logits, dim=1)
        loss = self.criterion(log_probs, labels)
        
        preds = torch.argmax(logits, dim=-1)
        target = torch.argmax(labels, dim=-1)

        self.val_acc(preds, target)
        self.val_f1(preds, target)

        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        pixel_values, labels = batch
        logits = self(pixel_values)
        log_probs = F.log_softmax(logits, dim=1)
        loss = self.criterion(log_probs, labels)
        
        preds = torch.argmax(logits, dim=-1)
        target = torch.argmax(labels, dim=-1)

        self.test_acc.update(preds, target) 
        self.test_f1.update(preds, target)
        self.conf_mat.update(preds, target)

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)        
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_test_epoch_end(self):
          if self.trainer.is_global_zero:
              print("\nConfusion Matrix:\n", self.conf_mat.compute().cpu().numpy())
          self.conf_mat.reset()
  

    def configure_optimizers(self):
          params = [{"params": self.classifier.parameters(), "lr": self.hparams.lr_head}]
          if not self.hparams.freeze_backbone:
              params.append({"params": self.backbone.parameters(), "lr": self.hparams.lr_backbone})
          return torch.optim.AdamW(params)



"""
Easy method : choose 1 annotator
positive 

Hard method : use 3 annotator data and make the prediction
1 positive = 0.33
2 positive = 0.67, neutral 0.33
3 positve  = 1


"""



if __name__ == "__main__":

    if not torch.cuda.is_available():
        raise RuntimeError("No GPU found. This script requires GPU.")

    devices = torch.cuda.device_count()
    accelerator = "gpu"
    strategy = "ddp" if devices > 1 else "auto"

    df = pd.read_csv("mvsa_image_soft_labels.txt", sep="\s+", header=0)
    data_list = []
    labels_list = []
    
    for _, row in df.iterrows():
        img_id = int(row['ID'])
        img_path = f"data/{img_id}.jpg"
        
        if os.path.exists(img_path):
            try: 
                with Image.open(img_path) as img:
                    img.convert("RGB")
                
                neg = float(row['neg'])
                neu = float(row['neu'])
                pos = float(row['pos'])
                total = neg + neu + pos
                
                if total == 0:
                    soft_label = [1/3, 1/3, 1/3]
                else:
                    soft_label = [neg / total, neu / total, pos / total]
                
                data_list.append((img_path, soft_label))
                hard_label = int(torch.argmax(torch.tensor(soft_label)))  
                labels_list.append(hard_label)      

            except Exception as e:
                print(f"Skipping broken image {img_path}: {e}")
        else:
            print(f"Image not found: {img_path}")


    train_data, temp_data, train_labels, temp_labels = train_test_split(data_list, labels_list, test_size=0.2, random_state=42, stratify=labels_list)
    val_data, test_data, val_labels, test_labels = train_test_split(temp_data, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels)
    print(f"Splits -> Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    selected_model = "openai/clip-vit-base-patch32"
    selected_model_safe_path = selected_model.replace("/", "-")
    processor = AutoProcessor.from_pretrained(selected_model, use_fast = True)

    train_dataset = MVSAImageDataset(train_data, processor)
    val_dataset = MVSAImageDataset(val_data, processor)
    test_dataset = MVSAImageDataset(test_data, processor)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=single_collate_fn, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=single_collate_fn, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=single_collate_fn, num_workers=4, pin_memory=True)

    model = VisionLanguageModel(model_name=selected_model, num_classes=3, freeze_backbone=True, lr_head=1e-3, lr_backbone=1e-6,)
    checkpoint_top3 = ModelCheckpoint(monitor="val_acc", mode="max", save_top_k=3, filename=selected_model_safe_path + "-{epoch:02d}-{val_acc:.4f}", dirpath="checkpoints/top3")
    checkpoint_last = ModelCheckpoint(save_last=True, filename=selected_model_safe_path + "-last", dirpath="checkpoints/last")
    csv_logger = CSVLogger("logs", name=f"{selected_model_safe_path}")


    trainer = pl.Trainer(max_epochs=50, accelerator=accelerator, devices=devices, strategy=strategy, precision="16-mixed", logger=csv_logger, callbacks=[checkpoint_top3, checkpoint_last] )
    ckpt_path = "checkpoints/last/last.ckpt" if os.path.exists("checkpoints/last/last.ckpt") else None  

    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
    trainer.test(model, test_loader)