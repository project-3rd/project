import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
np.bool = np.bool_  # numpy bool 호환성 문제 해결
import gluonnlp as nlp
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, vocab, max_len, pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, vocab=vocab, pad=pad, pair=pair)
        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return len(self.labels)

class BERTClassifier(nn.Module):
    def __init__(self, bert, hidden_size=768, num_classes=2, dr_rate=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        _, pooler = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long(), attention_mask=attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            pooler = self.dropout(pooler)
        return self.classifier(pooler)

class UnseenDataset(Dataset):
    def __init__(self, dataset, bert_tokenizer, vocab, max_len, pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, vocab=vocab, pad=pad, pair=pair)
        self.sentences = [transform([text]) for text in dataset]

    def __getitem__(self, i):
        return self.sentences[i]

    def __len__(self):
        return len(self.sentences)

def calc_accuracy(X, Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy() / max_indices.size()[0]
    return train_acc

def plot_metrics(train_accuracies, test_accuracies, train_losses, test_losses):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def train_model(model, train_dataloader, test_dataloader, num_epochs, optimizer, scheduler, loss_fn, log_interval, max_grad_norm, device):
    train_accuracies = []
    test_accuracies = []
    train_losses = []
    test_losses = []

    for e in range(num_epochs):
        train_acc = 0.0
        test_acc = 0.0
        train_loss = 0.0
        test_loss = 0.0
        model.train()
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            label = label.long().to(device)
            out = model(token_ids, valid_length, segment_ids)
            loss = loss_fn(out, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            train_acc += calc_accuracy(out, label)
            train_loss += loss.item()
            if batch_id % log_interval == 0:
                print(f"epoch {e+1} batch id {batch_id+1} loss {loss.data.cpu().numpy()} train acc {train_acc / (batch_id+1)}")
        train_accuracies.append(train_acc / (batch_id+1))
        train_losses.append(train_loss / (batch_id+1))
        print(f"epoch {e+1} train acc {train_acc / (batch_id+1)}")

        model.eval()
        with torch.no_grad():
            for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(test_dataloader)):
                token_ids = token_ids.long().to(device)
                segment_ids = segment_ids.long().to(device)
                label = label.long().to(device)
                out = model(token_ids, valid_length, segment_ids)
                loss = loss_fn(out, label)
                test_acc += calc_accuracy(out, label)
                test_loss += loss.item()
        test_accuracies.append(test_acc / (batch_id+1))
        test_losses.append(test_loss / (batch_id+1))
        print(f"epoch {e+1} test acc {test_acc / (batch_id+1)}")

    return train_accuracies, test_accuracies, train_losses, test_losses
