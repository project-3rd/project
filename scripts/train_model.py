import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
np.bool = np.bool_  # numpy bool 호환성 문제 해결
import gluonnlp as nlp
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel, AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from utils import BERTDataset, BERTClassifier, calc_accuracy, plot_metrics

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 토크나이저와 모델 로드
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
bertmodel = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)
vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token='[PAD]')

# 데이터셋 다운로드
!wget https://www.dropbox.com/s/374ftkec978br3d/ratings_train.txt?dl=1 -O data/ratings_train.txt
!wget https://www.dropbox.com/s/977gbwh542gdy94/ratings_test.txt?dl=1 -O data/ratings_test.txt

# 데이터셋 로드
dataset_train = nlp.data.TSVDataset("data/ratings_train.txt", field_indices=[1, 2], num_discard_samples=1)
dataset_test = nlp.data.TSVDataset("data/ratings_test.txt", field_indices=[1, 2], num_discard_samples=1)

# 데이터셋 크기 줄이기 (예제 용도)
subset_size = 20
dataset_train = dataset_train[:subset_size]
dataset_test = dataset_test[:subset_size]

# BERT를 위한 데이터 준비
max_len = 64
batch_size = 64
warmup_ratio = 0.1
num_epochs = 5
max_grad_norm = 1
log_interval = 200
learning_rate = 5e-5

data_train = BERTDataset(dataset_train, 0, 1, tokenizer.tokenize, vocab, max_len, True, False)
data_test = BERTDataset(dataset_test, 0, 1, tokenizer.tokenize, vocab, max_len, True, False)

train_dataloader = DataLoader(data_train, batch_size=batch_size, num_workers=5)
test_dataloader = DataLoader(data_test, batch_size=batch_size, num_workers=5)

# 모델 초기화
model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)

# 옵티마이저와 스케줄러 설정
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

# 모델 학습
train_accuracies, test_accuracies, train_losses, test_losses = train_model(
    model, train_dataloader, test_dataloader, num_epochs, optimizer, scheduler, loss_fn, log_interval, max_grad_norm, device
)

# 메트릭 시각화
plot_metrics(train_accuracies, test_accuracies, train_losses, test_losses)

# 혼동 행렬과 분류 보고서 생성
all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(test_dataloader)):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        _, preds = torch.max(out, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(label.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

print(classification_report(all_labels, all_preds, target_names=['Negative', 'Positive']))

# 모델 저장
model_save_path = "models/kobert_model999.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")
