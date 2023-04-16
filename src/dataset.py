import torch
from torch.utils.data import Dataset, DataLoader
from transformers import ( 
    AutoModelForSequenceClassification,
    AutoTokenizer,
    modelForSequenceClassification,
    get_constant_schedule_with_warmup,
)

import pytorch_lightning as pl
from torchmetrics.functional import accuracy, f1
from torchmetrics import Accuracy, F1, Precision, Recall
import wandb


class FinanciaSentimental(Dataset):
    """This class is used to load the data and tokenize it"""
    def __init__(self, tokenizer, data, type):
        self.tokenizer = tokenizer
        self.dataframe = dataframe
        ## Columns to target
        self._columns = ["target_sentiment", "companies_sentiment", "consumers_sentiment"]
    
    @property
    def columns(self):
        """Return the columns to target"""
        return self._columns

    def __len__(self):
        """Return the length of the dataset"""
        return self.dataframe.count()
        
    def __getitem__(self, index):
        """Get the data at the index"""
        values = self.dataframe.iloc[index]
        text = values['text']
        label = pd.get_dummies(values[self._columns], columns=[self._columns]).values.astype(np.int8)
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
        label = torch.tensor(label, dtype=torch.int8)
        return inputs['input_ids'].squeeze(0), inputs['attention_mask'].squeeze(0), label
    


class FinanciaMultilabel(pl.LightningModule):
    """This class is used to create the model"""
    def __init__(self, num_labels, model_name, class_weights):
        super().__init__()
        self.num_labels = num_labels
        # The models is multi-label, so we need to use BCEWithLogitsLoss
        self.loss = nn.BCEWithLogitsLoss(pos_weight=class_weights, reduction='none')
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=self.num_labels, ignore_mismatched_sizes=True)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        #Metrics for the model for training
        self.train_f1 = F1(task = "multilabel", num_classes=self.num_labels)
        self.train_accuracy = Accuracy(task ="multilabel", num_classes=self.num_labels)
        self.train_precision = Precision(task ="multilabel", num_classes=self.num_labels)
        self.train_recall = Recall(task ="multilabel", num_classes=self.num_labels)
        # Metrics for the model for validation
        self.val_f1 = F1(task = "multilabel", num_classes=self.num_labels)
        self.val_accuracy = Accuracy(task ="multilabel", num_classes=self.num_labels)
        self.val_precision = Precision(task ="multilabel", num_classes=self.num_labels)
        self.val_recall = Recall(task ="multilabel", num_classes=self.num_labels)
        
        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask):
        """This function is used to forward the data through the model"""
        output = self.model(input_ids, attention_mask=attention_mask)
        logits = output.logits
        return logits

    def training_step(self, batch, batch_idx):
        """This function is used to train the model"""
        input_ids, attention_mask, labels = batch
        logits = self(input_ids, attention_mask)
        train_loss = self.loss(logits, labels)
        train_acc = self.train_accuracy(logits, labels)
        train_f1 = self.train_f1(logits, labels)
        train_score = self.train_precision(logits, labels)
        train_recall = self.train_recall(logits, labels)
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        self.log("train/accuracy", self.train_accuracy, on_step=False, on_epoch=True)
        self.log("train/precision", self.train_precision, on_step=False, on_epoch=True)
        self.log("train/recall", self.train_recall, on_step=False, on_epoch=True)
        self.log("train/f1_score", self.train_f1_score, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """This function is used to validate the model"""
        input_ids, attention_mask, labels = batch
        logits = self(input_ids, attention_mask)
        val_loss = self.loss(logits, labels)
        val_acc = self.val_accuracy(outputs, labels)
        val_f1 = self.val_f1(outputs, labels)
        val_recall = self.val_recall(outputs, labels)
        val_precision = self.val_precision(outputs, labels)
        
        self.log("val/loss", val_loss, on_step=False, on_epoch=True)
        self.log("val/accuracy", self.val_accuracy, on_step=False, on_epoch=True)
        self.log("val/precision", self.val_precision, on_step=False, on_epoch=True)
        self.log("val/recall", self.val_recall, on_step=False, on_epoch=True)
        self.log("val/f1_score", self.val_f1_score, on_step=False, on_epoch=True)
        self.log("val/loss", val_loss, on_step=False, on_epoch=True)
        return val_loss

    def configure_optimizers(self):
        """This function is used to configure the optimizer"""
        optimizer = torch.optim.AdamW

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



    