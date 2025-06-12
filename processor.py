import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
import pickle
import os
import math
from typing import Optional, Callable


class FeatureExtractor:
    """Извлечение признаков из видеокадров с помощью ResNet152"""

    def __init__(self, cnn_model=None):
        if cnn_model is None:
            cnn_model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
            self.model = nn.Sequential(*list(cnn_model.children())[:-1])
        else:
            self.model = cnn_model

        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def extract_features(self, frames, device):
        """Извлекает признаки из списка кадров"""
        features = []

        with torch.no_grad():
            for frame in frames:
                frame = self.transform(frame).unsqueeze(0).to(device)
                feature = self.model(frame)
                feature = feature.squeeze()
                features.append(feature.cpu())

        return torch.stack(features)


class Encoder(nn.Module):
    """Видео-энкодер с двунаправленным LSTM"""

    def __init__(self, feature_dim, hidden_dim, num_layers=1, dropout=0.5):
        super(Encoder, self).__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

    def forward(self, features):
        if features.dim() > 3:
            batch_size, seq_len = features.size(0), features.size(1)
            features = features.view(batch_size, seq_len, -1)

        if features.size(-1) != self.feature_dim:
            raise ValueError(
                f"Неверная размерность признаков: ожидается {self.feature_dim}, получено {features.size(-1)}")

        outputs, hidden = self.lstm(features)
        return outputs, hidden


class MultiHeadAttention(nn.Module):
    """Multi-head attention механизм"""

    def __init__(self, encoder_dim, decoder_dim, num_heads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        assert decoder_dim % num_heads == 0

        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.num_heads = num_heads
        self.head_dim = decoder_dim // num_heads

        self.query_linear = nn.Linear(decoder_dim, decoder_dim)
        self.key_linear = nn.Linear(encoder_dim, decoder_dim)
        self.value_linear = nn.Linear(encoder_dim, decoder_dim)
        self.out_linear = nn.Linear(decoder_dim, encoder_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_outputs, decoder_hidden):
        batch_size, seq_len, _ = encoder_outputs.size()

        Q = self.query_linear(decoder_hidden.unsqueeze(1))
        K = self.key_linear(encoder_outputs)
        V = self.value_linear(encoder_outputs)

        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, 1, self.decoder_dim
        ).squeeze(1)

        context = self.out_linear(context)
        avg_attn_weights = attn_weights.mean(dim=1).squeeze(1)

        return context, avg_attn_weights


class Decoder(nn.Module):
    """Декодер с механизмом внимания"""

    def __init__(self, vocab_size, embed_dim, encoder_dim, hidden_dim, attention_dim, dropout=0.5):
        super(Decoder, self).__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.encoder_dim = encoder_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = MultiHeadAttention(encoder_dim, hidden_dim, num_heads=8, dropout=dropout)
        self.lstm = nn.LSTMCell(embed_dim + encoder_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

        self.init_weights()

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, batch_size, device):
        h = torch.zeros(batch_size, self.hidden_dim).to(device)
        c = torch.zeros(batch_size, self.hidden_dim).to(device)
        return h, c

    def beam_search(self, encoder_outputs, vocab, max_length=20, beam_size=5):
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device

        h, c = self.init_hidden_state(batch_size, device)

        start_idx = vocab.word2idx['<start>']
        end_idx = vocab.word2idx['<end>']

        beams = [(0.0, [start_idx], h, c)]
        completed = []

        for _step in range(max_length):
            all_candidates = []

            for score, seq, h, c in beams:
                if seq[-1] == end_idx:
                    completed.append((score, seq))
                    continue

                last = torch.LongTensor([seq[-1]]).to(device)

                emb = self.embedding(last)
                context, _ = self.attention(encoder_outputs, h)
                lstm_input = torch.cat([emb, context], dim=1)
                h_new, c_new = self.lstm(lstm_input, (h, c))
                out = self.fc(h_new)
                logp = F.log_softmax(out, dim=1)

                topk_logp, topk_idx = logp.topk(beam_size, dim=1)

                for k in range(beam_size):
                    next_token = topk_idx[0, k].item()
                    next_score = topk_logp[0, k].item()

                    new_score = score + next_score
                    new_seq = seq + [next_token]
                    all_candidates.append((new_score, new_seq, h_new.clone(), c_new.clone()))

            if not all_candidates:
                break

            all_candidates.sort(key=lambda x: x[0], reverse=True)
            beams = all_candidates[:beam_size]

            if all(b[1][-1] == end_idx for b in beams):
                completed.extend(beams)
                break

        if not completed:
            completed = beams

        completed.sort(key=lambda x: x[0], reverse=True)
        best_seq = completed[0][1]

        if best_seq and best_seq[0] == start_idx:
            best_seq = best_seq[1:]

        if end_idx in best_seq:
            cut = best_seq.index(end_idx)
            best_seq = best_seq[:cut]

        best_seq = [tok for tok in best_seq if tok != end_idx]

        return torch.LongTensor(best_seq).to(device)


class VideoCaptioningModel(nn.Module):
    """Полная модель генерации подписей к видео"""

    def __init__(self, vocab_size, feature_dim=2048, embed_dim=512, encoder_dim=512,
                 decoder_dim=512, attention_dim=512, dropout=0.5):
        super(VideoCaptioningModel, self).__init__()

        self.encoder = Encoder(
            feature_dim=feature_dim,
            hidden_dim=encoder_dim,
            dropout=dropout
        )

        self.decoder = Decoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            encoder_dim=encoder_dim * 2,
            hidden_dim=decoder_dim,
            attention_dim=attention_dim,
            dropout=dropout
        )

class Vocabulary:
    """Класс словаря для преобразования токенов"""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def __call__(self, word):
        return self.word2idx.get(word, self.word2idx.get('<unk>', 0))

    def __len__(self):
        return len(self.word2idx)


class VideoProcessor:
    """Основной класс для обработки видео и генерации описаний"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.vocab = None
        self.feature_extractor = None
        self.model_loaded = False

    def load_model(self):
        """Загрузка модели и словаря из файлов"""
        try:
            model_path = "resources/final_model.pt"
            vocab_path = "resources/vocabulary.pkl"

            # Проверяем существование файлов
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Файл модели не найден: {model_path}")
            if not os.path.exists(vocab_path):
                raise FileNotFoundError(f"Файл словаря не найден: {vocab_path}")

            # Загружаем словарь
            with open(vocab_path, 'rb') as f:
                self.vocab = pickle.load(f)

            # Инициализируем модель с параметрами из обучения
            self.model = VideoCaptioningModel(
                vocab_size=len(self.vocab),
                feature_dim=2048,
                embed_dim=512,
                encoder_dim=512,
                decoder_dim=512,
                attention_dim=512,
                dropout=0.3
            ).to(self.device)

            # Загружаем веса модели
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state'])
            self.model.eval()

            # Инициализируем экстрактор признаков
            self.feature_extractor = FeatureExtractor()
            self.feature_extractor.model = self.feature_extractor.model.to(self.device)

            self.model_loaded = True
            print(f"Модель успешно загружена на устройство: {self.device}")

        except Exception as e:
            print(f"Ошибка при загрузке модели: {str(e)}")
            self.model_loaded = False
            raise

    def extract_frames(self, video_path, max_frames=40, progress_callback=None):
        """Извлечение кадров из видео"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Не удается открыть видео: {video_path}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_rate = max(1, frame_count // max_frames)
        frames = []

        for i in range(0, frame_count, sample_rate):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                # Обновляем прогресс
                if progress_callback:
                    progress = min(i / frame_count, 1.0)
                    progress_callback("extract_frames", progress)

            if len(frames) >= max_frames:
                break

        cap.release()

        # Дополняем до нужного количества кадров
        if len(frames) < max_frames and len(frames) > 0:
            last_frame = frames[-1]
            frames.extend([last_frame for _ in range(max_frames - len(frames))])
        elif len(frames) == 0:
            frames = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(max_frames)]

        return frames

    def predict_caption(self, features, beam_size=10, max_length=20):
        """Генерация описания с использованием beam search"""
        if not self.model_loaded:
            raise RuntimeError("Модель не загружена")

        self.model.eval()

        with torch.no_grad():
            # Прямой проход через энкодер
            enc_out, _ = self.model.encoder(features)

            # Генерация описания
            predicted_sequence = self.model.decoder.beam_search(
                enc_out, self.vocab, max_length=max_length, beam_size=beam_size
            )

            # Преобразование в слова
            words = []
            for idx in predicted_sequence.cpu().numpy():
                if idx in self.vocab.idx2word:
                    word = self.vocab.idx2word[idx]
                    if word not in ['<start>', '<end>', '<pad>']:
                        words.append(word)

        return ' '.join(words)

    def process_video(self, video_path, progress_callback=None):
        """Основная функция обработки видео"""
        try:
            if not self.model_loaded:
                raise RuntimeError("Модель не загружена. Вызовите load_model() сначала.")

            # Этап 1: Извлечение кадров
            if progress_callback:
                progress_callback("extract_frames", 0.0)

            frames = self.extract_frames(video_path, progress_callback=progress_callback)

            # Этап 2: Извлечение признаков
            if progress_callback:
                progress_callback("extract_features", 0.0)

            features = self.feature_extractor.extract_features(frames, self.device)

            # Добавляем размерность батча
            features = features.unsqueeze(0).to(self.device)

            if progress_callback:
                progress_callback("extract_features", 1.0)

            # Этап 3: Генерация описания
            if progress_callback:
                progress_callback("generate", 0.0)

            description = self.predict_caption(features)

            if progress_callback:
                progress_callback("generate", 1.0)

            return description

        except Exception as e:
            raise Exception(f"Ошибка при обработке видео: {str(e)}")
