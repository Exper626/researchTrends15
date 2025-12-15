import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassConfusionMatrix
from transformers import AutoTokenizer, AutoModel
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from functools import partial


os.environ["TOKENIZERS_PARALLELISM"] = "false"

class T4SASentimentDataset(Dataset):

    def __init__(self, data, tokenizer, max_token_length=128):
        self.tokenizer = tokenizer
        self.max_token_length = max_token_length
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, labels = self.data[idx]
        encoding = self.tokenizer(
            text, truncation=True, max_length=self.max_token_length, padding=False, return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].flatten(), # Remove batch dim added by tokenizer
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(labels, dtype=torch.float32)
        }

def text_collate_fn(batch, pad_token_id=0):
    input_ids = [item["input_ids"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]
    labels = torch.stack([item["labels"] for item in batch])
    
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    attention_masks_padded = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)
    
    return input_ids_padded, attention_masks_padded, labels


class EncoderOnlyTransformer(pl.LightningModule):

    def __init__(self, model_name="bert-base-uncased", num_classes=3, lr_head=1e-3, lr_backbone=1e-5, freeze_backbone=False):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = AutoModel.from_pretrained(model_name)
        
        if hasattr(self.backbone.config, "hidden_size"):
            embed_dim = self.backbone.config.hidden_size

        elif hasattr(self.backbone.config, "d_model"): 
            embed_dim = self.backbone.config.d_model
        else:
            embed_dim = 768 

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

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None: # Strategy: Use Pooler Output if available, otherwise Extract CLS token
            cls_token_feature_vector = outputs.pooler_output
        else:           
            cls_token_feature_vector = outputs.last_hidden_state[:, 0, :]  # Taking the CLS token (index 0) from the last hidden state
       
        logits = self.classifier(cls_token_feature_vector)
        return logits
    




    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        logits = self(input_ids, attention_mask)        
        loss = self.criterion(F.log_softmax(logits, dim=1), labels)
        preds = torch.argmax(logits, dim=-1)
        target = torch.argmax(labels, dim=-1)
 
        self.log("train_acc", self.train_acc(preds, target), on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        logits = self(input_ids, attention_mask)
        
        log_probs = F.log_softmax(logits, dim=1)
        loss = self.criterion(log_probs, labels)
        
        preds = torch.argmax(logits, dim=-1)
        target = torch.argmax(labels, dim=-1)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc(preds, target), on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1",  self.val_f1(preds, target), on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        pass


    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        logits = self(input_ids, attention_mask)
        loss = self.criterion(F.log_softmax(logits, dim=1), labels)
        
"""
        Model prediction = logits

        ground truth = label


        why cant cross entrophy loss because we train on soft labels (ex: 0.343, 0.3434)

        to calculate loss therefore we use KL divergence

        to get it has to be exactly 1 
        thereofore logits has to be 1 we use log_softmax for that!!!

ok?
"""
        preds = torch.argmax(logits, dim=-1)
        target = torch.argmax(labels, dim=-1)

        self.log("test_loss", loss)
        self.log("test_acc", self.test_acc(preds, target), on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_f1", self.test_f1(preds, target), on_step=False, on_epoch=True, prog_bar=True)
        self.conf_mat.update(preds, target)
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


if __name__ == "__main__":

    if not torch.cuda.is_available():
        raise RuntimeError("No GPU found. This script requires GPU.")

    devices = torch.cuda.device_count()
    accelerator = "gpu"
    strategy = "ddp" if devices > 1 else "auto" 
    
    csv_file = "/kaggle/working/t4sa_merged_text_sentiment.csv"
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Cannot find {csv_file}")

    df = pd.read_csv(csv_file, header=0) 
        
    data_list = []        # Will hold (input_feature, soft_label)
    labels_list = []      # Will hold hard_label (int) for stratification
    skipped_count = 0



    for _, row in df.iterrows():
        try:
            input_feature = str(row['content']) 
            neg = float(row['NEG'])
            neu = float(row['NEU'])
            pos = float(row['POS'])
            
            total = neg + neu + pos
      
            if total > 0:
                soft_label = [neg / total, neu / total, pos / total]
            else:
                soft_label = [1/3, 1/3, 1/3]  # Fallback: uniform distribution if all zeros (very rare)
            
            hard_label = int(torch.argmax(torch.tensor(soft_label))) 
            
            data_list.append((input_feature, soft_label))
            labels_list.append(hard_label)
            
        except Exception as e:
            skipped_count += 1
            print(f"Skipping row: {e}") 

    print(f"Loaded {len(data_list)} samples, skipped {skipped_count}")


    train_data, temp_data, _, temp_labels = train_test_split(
        data_list, labels_list, test_size=0.4, random_state=42, stratify=labels_list
    )
    val_data, test_data, val_labels, test_labels = train_test_split(
        temp_data, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    print(f"Splits -> Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    selected_model = "bert-base-uncased" 
    selected_model_safe_path = selected_model.replace("/", "-")
    tokenizer = AutoTokenizer.from_pretrained(selected_model, use_fast=True)
    text_collate_fn_safe_pad = partial(text_collate_fn, pad_token_id=tokenizer.pad_token_id)



    model = EncoderOnlyTransformer(
        model_name=selected_model,  
        num_classes=3,
        freeze_backbone=False,
        lr_head=1e-3,
        lr_backbone=2e-5, # Standard BERT LR is usually lower (2e-5 to 5e-5)
    )

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.backbone.resize_token_embeddings(len(tokenizer))


    train_dataset = T4SASentimentDataset(train_data, tokenizer)
    val_dataset   = T4SASentimentDataset(val_data, tokenizer)
    test_dataset  = T4SASentimentDataset(test_data, tokenizer)
    
    num_workers = os.cpu_count() // 2

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  collate_fn=text_collate_fn_safe_pad, num_workers=num_workers, pin_memory=True, persistent_workers=True, )
    val_loader   = DataLoader(val_dataset,  batch_size=32,   shuffle=False, collate_fn=text_collate_fn_safe_pad, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=32,  shuffle=False, collate_fn=text_collate_fn_safe_pad, num_workers=num_workers, pin_memory=True)


    checkpoint_top3 = ModelCheckpoint(monitor="val_acc", mode="max", save_top_k=3, 
        filename=selected_model_safe_path + "-{epoch:02d}-{val_acc:.4f}", 
        dirpath="checkpoints/text_top3"
    )

    checkpoint_last = ModelCheckpoint(save_last=True,  filename=selected_model_safe_path + "-last", dirpath="checkpoints/text_last")
    csv_logger = CSVLogger("logs_text", name=f"{selected_model_safe_path}")

    trainer = pl.Trainer(
        max_epochs=20, 
        accelerator=accelerator,
        devices=devices,         
        strategy=strategy,
        precision="16-mixed",
        logger=csv_logger,  
        callbacks=[checkpoint_top3, checkpoint_last] 
    )

    ckpt_path = "checkpoints/last/last.ckpt" if os.path.exists("checkpoints/last/last.ckpt") else None
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
    trainer.test(model, dataloaders=test_loader, ckpt_path=checkpoint_top3.best_model_path)