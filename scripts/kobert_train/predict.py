#예측

text
이 영화 정말 재미있어요
별로였어요. 다시 보고 싶지 않아요
기대 이상이었어요. 꼭 보세요!
...

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
np.bool = np.bool_
import gluonnlp as nlp
from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel
import pandas as pd

# GPU 사용 시
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# KoBERT 모델 및 토크나이저 불러오기
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
bertmodel = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)
vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token='[PAD]')

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, bert_tokenizer, vocab, max_len, pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, vocab=vocab, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]

    def __getitem__(self, i):
        return self.sentences[i]

    def __len__(self):
        return len(self.sentences)

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

# 학습된 모델 불러오기
model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)
model.load_state_dict(torch.load('/content/drive/MyDrive/hans_kobert_model.pth'))
model.eval()

# 새로운 데이터셋 불러오기 (예: CSV 파일로부터)
# new_data = pd.read_csv('new_dataset.csv')  # 데이터셋은 CSV 파일로 저장되어 있어야 합니다.
dataset_new = BERTDataset(new_data.values, 0, tokenizer.tokenize, vocab, max_len=64, pad=True, pair=False)
new_dataloader = DataLoader(dataset_new, batch_size=64, num_workers=5)

# 예측 수행
all_preds = []
with torch.no_grad():
    for batch_id, (token_ids, valid_length, segment_ids) in enumerate(new_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length = valid_length
        out = model(token_ids, valid_length, segment_ids)
        all_preds.extend(torch.argmax(out, 1).cpu().numpy())

# 예측 결과 저장
new_data['predictions'] = all_preds
# new_data.to_csv('predictions.csv', index=False)
print(new_data)
