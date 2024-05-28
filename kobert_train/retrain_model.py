#재학습

id\tdocument\tlabel
0\t이 영화 정말 재미있어요\t1
1\t이 영화 정말 지루해요\t0
2\t이 영화는 그저 그래요\t1
...


import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
np.bool = np.bool_
import gluonnlp as nlp
from tqdm import tqdm
from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel, AdamW, get_cosine_schedule_with_warmup
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# KoBERT 모델 및 토크나이저 불러오기
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
bertmodel = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)
vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token='[PAD]')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

데이터셋 불러오기 및 전처리
train_file_path = "path_to_your_train_dataset.tsv"
test_file_path = "path_to_your_test_dataset.tsv"

dataset_train = nlp.data.TSVDataset(train_file_path, field_indices=[1, 2], num_discard_samples=1)
dataset_test = nlp.data.TSVDataset(test_file_path, field_indices=[1, 2], num_discard_samples=1)

max_len = 64
batch_size = 64
tok = tokenizer.tokenize
data_train = BERTDataset(dataset_train, 0, 1, tok, vocab, max_len, True, False)
data_test = BERTDataset(dataset_test, 0, 1, tok, vocab, max_len, True, False)
train_dataloader = DataLoader(data_train, batch_size=batch_size, num_workers=5)
test_dataloader = DataLoader(data_test, batch_size=batch_size, num_workers=5)

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
        _, pooler = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long(),
                              attention_mask=attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)
model.load_state_dict(torch.load('/content/drive/MyDrive/jjj_kobert_model.pth')) # 경로

# 옵티마이저 및 스케줄러 설정
learning_rate = 5e-5
num_epochs = 5
warmup_ratio = 0.1
max_grad_norm = 1

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()
t_total = len(train_dataloader) * num_epochs
warmup_step = int(t_total * warmup_ratio)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

def calc_accuracy(X, Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy() / max_indices.size()[0]
    return train_acc

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
        valid_length = valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        loss = loss_fn(out, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        train_acc += calc_accuracy(out, label)
        train_loss += loss.item()
    train_accuracies.append(train_acc / len(train_dataloader))
    train_losses.append(train_loss / len(train_dataloader))
    print(f"epoch {e + 1} train acc {train_acc / len(train_dataloader)}")

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(test_dataloader)):
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length = valid_length
            label = label.long().to(device)
            out = model(token_ids, valid_length, segment_ids)
            loss = loss_fn(out, label)
            test_acc += calc_accuracy(out, label)
            test_loss += loss.item()
            all_preds.extend(torch.argmax(out, 1).cpu().numpy())
            all_labels.extend(label.cpu().numpy())
    test_accuracies.append(test_acc / len(test_dataloader))
    test_losses.append(test_loss / len(test_dataloader))
    print(f"epoch {e + 1} test acc {test_acc / len(test_dataloader)}")
    print(confusion_matrix(all_labels, all_preds))
    print(classification_report(all_labels, all_preds))

# 학습 결과 시각화
plt.figure(figsize=(12, 6))
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Test Accuracy')
plt.show()

# 손실 시각화
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Test Loss')
plt.show()

# 혼동 행렬 시각화
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# 학습된 모델 저장
torch.save(model.state_dict(), '/content/drive/MyDrive/jjj_kobert_model.pth') # 경로
