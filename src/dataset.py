import os
import pandas as pd

import torch
from torch.utils.data import Dataset

from src.kobert_tokenizer import KoBertTokenizer
import torchaudio
from transformers import Wav2Vec2Processor,Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor
import numpy as np

class KEMDyDataset(Dataset):
    def __init__(self, modal='text', k=1, kind='train', l_type='valence'):
        self.modal = modal
        self.k = k
        self.kind = kind
        self.l_type = l_type
        self.category_dict = {'neutral': 0, 'happy': 1, 'surprise': 2, 'angry': 3, 'sad': 4, 'disqust': 5, 'fear': 6}

        self.path = '../KEMDy20_v1_1/wav'
        self.label_path = '../KEMDy20_v1_1/annotation_e'

        self.data = None
        self.mask = None

        self.w_data = None
        self.w_mask = None

        self.label = None

        self.sess_seg = []

        if not modal == 'wav':
            self.tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')

        if not modal == 'text':
            model_name_or_path = "kresnik/wav2vec2-large-xlsr-korean"
            self.wav_tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_name_or_path)
            self.wav_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path)
            # self.processor = Wav2Vec2Processor(feature_extractor=self.wav_feature_extractor, tokenizer=self.wav_tokenizer)
            self.processor = Wav2Vec2Processor.from_pretrained(model_name_or_path)
            self.target_sampling_rate = self.processor.feature_extractor.sampling_rate
            print(f"The target sampling rate: {self.target_sampling_rate}")

        # Session Validation
        if kind == 'train':
            index = list(range(1, 41))
            del index[8*(k-1):8*k]
            self.load_data(index)
        else:
            index = list(range(8*(k-1)+1, 8*k+1))
            self.load_data(index)

    def load_data(self, index):
        sound = []
        sentence = []
        labels = []

        for session_file in os.listdir(self.label_path):
            session = session_file.split('_')[1]
            session_number = int(session[4:])
            if session_number in index:
                label_dataframe = pd.read_csv(f'{self.label_path}/{session_file}', index_col=0, header=0)
                session = session[:4] + 'ion' + session[4:]
                for i, row in label_dataframe.iterrows():
                    self.sess_seg.append((session_number, i))
                    segment = row['Segment ID']

                    if self.l_type == 'valence':
                        valence = round(float(row['Valence']))
                        labels.append(valence)
                    elif self.l_type == 'arousal':
                        arousal = round(float(row['Arousal']))
                        labels.append(arousal)
                    elif self.l_type == 'emotion':
                        emotion = row['Emotion']
                        labels.append(self.category_dict.get(emotion))

                    # Load Text
                    if self.modal == 'text':
                        with open(f'{self.path}/{session}/{segment}.txt', 'r', encoding='cp949') as file:
                            sentence.append(file.readline().strip())

                    # Load Audio ...
                    if self.modal == 'wav':
                        sound.append(self.speech_file_to_array_fn(f'{self.path}/{session}/{segment}.wav'))
                    
                    if self.modal == 'both':
                        with open(f'{self.path}/{session}/{segment}.txt', 'r', encoding='cp949') as file:
                            sentence.append(file.readline().strip())
                        sound.append(self.speech_file_to_array_fn(f'{self.path}/{session}/{segment}.wav'))


        self.label = torch.tensor(labels)

        if self.modal == 'text':
            self.data, self.mask = self.tokenize(sentence)
            print(self.data.shape, self.mask.shape, self.label.shape)
        
        if self.modal == 'wav':
            # results = self.wav_feature_extractor(sound, sampling_rate=self.target_sampling_rate, padding='max_length', return_attention_mask=True)
            results = self.processor(sound, sampling_rate=self.target_sampling_rate, padding="max_length", max_length=45520, truncation=True)
            self.w_data = torch.tensor(results['input_values'])
            self.w_mask = torch.tensor(results['attention_mask'])
            print(self.w_data.shape, self.w_mask.shape, self.label.shape)
        
        if self.modal == 'both':
            self.data, self.mask = self.tokenize(sentence)
            print(self.data.shape, self.mask.shape, self.label.shape)
            
            results = self.processor(sound, sampling_rate=self.target_sampling_rate, padding="max_length", max_length=45520, truncation=True)
            self.w_data = torch.tensor(np.array(results['input_values']))
            self.w_mask = torch.tensor(np.array(results['attention_mask']))
            print(self.w_data.shape, self.w_mask.shape, self.label.shape)


    def tokenize(self, sentence):
        encode = self.tokenizer.batch_encode_plus(sentence, truncation=True, padding='max_length')

        return torch.tensor(encode['input_ids']), torch.tensor(encode['attention_mask'])

    def __getitem__(self, index):
        if self.modal == 'text':
            data, mask, label = self.data[index], self.mask[index], self.label[index]
            return (data, mask), label
        
        if self.modal == 'wav':
            data, mask, label = self.w_data[index], self.w_mask[index], self.label[index]
            return (data, mask), label
        
        if self.modal == 'both':            
            data, mask, label = self.data[index], self.mask[index], self.label[index]
            w_data, w_mask, label = self.w_data[index], self.w_mask[index], self.label[index]

            return (data, mask, w_data, w_mask), label
    
    def speech_file_to_array_fn(self, path):
        speech_array, sampling_rate = torchaudio.load(path)
        resampler = torchaudio.transforms.Resample(sampling_rate, self.target_sampling_rate)
        speech = resampler(speech_array).squeeze().numpy()
        return speech

    def __len__(self):
        if self.modal == 'text' or self.modal=='both':
            return self.data.size(0)
        else:
            return self.w_data.size(0)
