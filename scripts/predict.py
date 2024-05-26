import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
np.bool = np.bool_  # numpy bool 호환성 문제 해결
import gluonnlp as nlp
import pandas as pd
from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel
from utils import BERTClassifier, UnseenDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 토크나이저와 모델 로드
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
bertmodel = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)
vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token='[PAD]')

# 모델 초기화 및 로드
model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)
model_save_path = "models/kobert_model_retrained.pth"
model.load_state_dict(torch.load(model_save_path))

# 예측 함수 정의
def predict(text_list, model, tokenizer, vocab, max_len, batch_size, device):
    model.eval()
    data = UnseenDataset(text_list, tokenizer, vocab, max_len, True, False)
    dataloader = DataLoader(data, batch_size=batch_size, num_workers=0)
    predictions = []
    with torch.no_grad():
        for batch_id, (token_ids, valid_length, segment_ids) in enumerate(dataloader):
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length = valid_length
            out = model(token_ids, valid_length, segment_ids)
            max_vals, max_indices = torch.max(out, 1)
            predictions.extend(max_indices.cpu().numpy())
    return predictions

# 예측을 위한 새로운 데이터 생성
data = {
    'text': ["이 영화 정말 재미있어요", "별로 좋지 않았어요", "훌륭한 연기와 스토리", "지루하고 시간 낭비에요", "감동적인 이야기였어요"]
}
unseen_dataset = pd.DataFrame(data)

# 예측 수행
max_len = 64
batch_size = 64
unseen_texts = unseen_dataset['text'].tolist()
predictions = predict(unseen_texts, model, tokenizer, vocab, max_len, batch_size, device)
unseen_dataset['predictions'] = predictions

print(unseen_dataset)
