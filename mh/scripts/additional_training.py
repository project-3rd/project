import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
np.bool = np.bool_  # numpy bool 호환성 문제 해결
import gluonnlp as nlp
from tqdm import tqdm
import pandas as pd
from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel, AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from utils import BERTDataset, BERTClassifier, calc_accuracy, plot_metrics

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 토크나이저와 모델 로드
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
bertmodel = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)
vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token='[PAD]')

# 새로운 학습 데이터 생성
train_data = {
    'text': [
        '이 영화 정말 재미있어요', '별로 좋지 않았어요', '정말 최고의 영화입니다', '다시 보고 싶지 않아요',
        '훌륭한 연기와 멋진 스토리', '지루하고 재미없었어요', '감동적인 영화였습니다', '내용이 너무 평범했어요'
    ],
    'label': [1, 0, 1, 0, 1, 0, 1, 0]
}
train_dataset = pd.DataFrame(train_data)
train_dataset.to_csv('data/new_train_dataset.tsv', sep='\t', index=False)
dataset_train = nlp.data.TSVDataset("data/new_train_dataset.tsv", field_indices=[0, 1], num_discard_samples=1)

# BERT를 위한 데이터 준비
max_len = 64
batch_size = 64
warmup_ratio = 0.1
num_epochs = 5
max_grad_norm = 1
log_interval = 200
learning_rate = 5e-5

data_train = BERTDataset(dataset_train, 0, 1, tokenizer.tokenize, vocab, max_len, True, False)
train_dataloader = DataLoader(data_train, batch_size=batch_size, num_workers=5)

# 새로운 테스트 데이터 생성
test_data = {
    'text': [
        '이 영화 최고예요', '정말 별로였어요', '다시 보고 싶은 영화입니다', '시간 낭비였어요',
        '스토리가 훌륭해요', '재미없고 지루해요', '매우 감동적인 영화였어요', '볼 가치가 없어요'
    ],
    'label': [1, 0, 1, 0, 1, 0, 1, 0]
}
test_dataset = pd.DataFrame(test_data)
test_dataset.to_csv('data/new_test_dataset.tsv', sep='\t', index=False)
dataset_test = nlp.data.TSVDataset("data/new_test_dataset.tsv", field_indices=[0, 1], num_discard_samples=1)

data_test = BERTDataset(dataset_test, 0, 1, tokenizer.tokenize, vocab, max_len, True, False)
test_dataloader = DataLoader(data_test, batch_size=batch_size, num_workers=5)

# 모델 초기화 및 로드
model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)
model_save_path = "models/kobert_model999.pth"
model.load_state_dict(torch.load(model_save_path))

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
model_save_path = "models/kobert_model_retrained.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")
