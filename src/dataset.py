from sklearn.metrics import confusion_matrix
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification

import pytorch_lightning as pl
from torchmetrics.functional import accuracy, f1
import wandb


class FinanciaSentimental(Dataset):
    """This class is used to load the data and tokenize it"""
    def __init__(self, tokenizer, data, type):
        self.tokenizer = tokenizer
        self.dataframe = dataframe
        
    def __len__(self):
        """Return the length of the dataset"""
        return self.dataframe.count()
        
    def __getitem__(self, index):
        """Get the data at the index"""
        values = self.dataframe.iloc[index]
        label = values[type]
        text = values['text']
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return inputs['input_ids'].squeeze(0), inputs['attention_mask'].squeeze(0), label
    

class BERTClassifier(pl.LightningModule):
    """This class is used to create the model"""
    def __init__(self, labels, model_name):
        super().__init__()
        self.labels = labels
        self.bert = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(labels)))
        
    def forward(self, input_ids, attention_mask):
        """This function is used to forward the data through the model"""
        return self.bert(input_ids, attention_mask=attention_mask)[0]
        
    def training_step(self, batch, batch_idx):
        """This function is used to train the model"""
        input_ids, attention_mask, labels = batch
        outputs = self(input_ids, attention_mask)
        predictions = torch.argmax(outputs, dim=1)
        train_loss = self.loss(outputs, labels)
        train_acc = accuracy(predictions, labels)
        train_f1 = f1(predictions, labels, num_classes=self.num_labels, average='macro')
        wandb.log({'train_loss': train_loss, 'train_acc': train_acc, 'train_f1': train_f1})
        return {'loss': train_loss}
        
    def validation_step(self, batch, batch_idx):
        """This function is used to validate the model"""
        input_ids, attention_mask, labels = batch
        outputs = self(input_ids, attention_mask)
        predictions = torch.argmax(outputs, dim=1)
        val_loss = self.loss(outputs, labels)
        val_acc = accuracy(predictions, labels)
        val_f1 = f1(predictions, labels, num_classes=self.num_labels, average='macro')
        wandb.log({'val_loss': val_loss, 'val_acc': val_acc, 'val_f1': val_f1})
        return {'val_loss': val_loss}

    def test_epoch_end(self, outputs):
        """This function is used to test the model"""
        test_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        test_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        test_f1 = torch.stack([x['test_f1'] for x in outputs]).mean()
        wandb.log({'test_loss': test_loss, 'test_acc': test_acc, 'test_f1': test_f1})
        
    def configure_optimizers(self):
        """This function is used to configure the optimizer"""
        optimizer = torch.optim.AdamW(self.parameters(), lr=2e-5)
        return optimizer

"""
model = BERTClassifier('bert-base-uncased', num_classes=2)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_data = IMDBDataset(tokenizer, train_texts, train_labels)
val_data = IMDBDataset(tokenizer, val_texts, val_labels)
test_data = IMDBDataset(tokenizer, test_texts, test_labels)
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
val_loader = DataLoader(val_data, batch_size=8)
test_loader = DataLoader(test_data, batch_size=8)
wandb.init(project='my-project-name', config={
    'batch_size': 8,
    'learning_rate': 2e-5
})
trainer = pl.Trainer(gpus=1, max_epochs=10)
trainer.fit(model, train_loader, val_loader)
trainer.test(model, test_loader)
wandb.log({'test_loss': trainer.callback_metrics['test_loss'].item()})
"""



    