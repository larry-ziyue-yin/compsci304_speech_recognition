# Load audio and extract Mel spectrogram
import numpy as np
from scipy.io import wavfile
import whisper
import torch
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, tokenizer, model_name, max_length, lang):
        self.path=data_path
        self.data_list=[]
        with open(self.path,'r') as f:
            for line in f:
                audio_path,text=line.split('	')
                self.data_list.append([audio_path,text])
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.max_length = max_length
        self.n_mels = 128 if 'large-v3' in model_name else 80
        self.special_tokens = set(self.tokenizer.special_tokens.values())
        self.lang=lang
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        #=========================TODO===================
        #read audio_path, text from self.data_list
        #extract audio_feature using wavfile.read function
        audio_path, text = self.data_list[idx]
        audio_path = audio_path.strip()
        text = text.strip()
        _, wav_data = wavfile.read(audio_path)

        #=======================TODO=====================
        if wav_data.ndim > 1:
            wav_data = wav_data.mean(axis=1)

        if np.issubdtype(wav_data.dtype, np.integer):
            scale = float(np.iinfo(wav_data.dtype).max)
            audio = wav_data.astype(np.float32) / scale
        else:
            audio = wav_data.astype(np.float32)

        audio = audio.flatten()
        key = audio_path.split('/')[-1][:-4]
        # Padding or truncation
        if self.max_length is not None:
            audio = whisper.pad_or_trim(audio, length=self.max_length)
        
        # extract Mel spectrogram
        mel = whisper.log_mel_spectrogram(audio, n_mels=self.n_mels)
        
        
        # Construct decoder inputs and labels
        dec_input_ids = [
            self.tokenizer.sot, 
            self.tokenizer.special_tokens.get(f"<|{self.lang}|>", self.tokenizer.sot),
            self.tokenizer.transcribe, 
            self.tokenizer.no_timestamps
        ] + self.tokenizer.encode(" " + text.strip())
        
        labels = dec_input_ids[1:] + [self.tokenizer.eot]
        
        return {
            "input_ids": mel,  #  [n_mels, time]
            "labels": torch.LongTensor(labels),
            "dec_input_ids": torch.LongTensor(dec_input_ids),
            "name": key
        }
