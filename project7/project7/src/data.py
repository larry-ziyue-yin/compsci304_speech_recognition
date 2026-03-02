import torch
from functools import partial
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from .wtimit import Dataset
import whisper
import numpy as np
def WhisperDataCollatorWhithPadding(features):
    input_ids, labels, dec_input_ids, names = [], [], [], []
    for f in features:
        input_ids.append(f["input_ids"])
        labels.append(f["labels"])
        dec_input_ids.append(f["dec_input_ids"])
        names.append(f["name"])

    audio_lengths = [audio.shape[1] for audio in input_ids]
    max_audio_len =  max(audio_lengths)
    input_ids = [np.pad(audio, ((0, 0), (0, max_audio_len - audio_len)), 'constant', constant_values=0) for audio, audio_len in zip(input_ids, audio_lengths)]

    label_lengths = [len(lab) for lab in labels]
    dec_input_ids_length = [len(e) for e in dec_input_ids]
    max_label_len = max(label_lengths+dec_input_ids_length) # seems redundant

    #==============================TODO===============================
    # pad labels and dec_input_ids (np.pad)
    # pad the labels with -100 (dummy, ignore index in cross-entropy), pad the dec_input_ids with eot 50257

    #==============================TODO===============================
    batch = {
        "input_ids": input_ids,
        "labels": labels,
        "dec_input_ids": dec_input_ids,
    }

    batch = {k: torch.tensor(np.array(v), requires_grad=False) for k, v in batch.items()}

    batch["name"] = names
    return batch
    

def load_dataset_wtimit(n_jobs, use_gpu, pin_memory, config):
    ''' Data loading function specifically for Whisper fine-tuning '''
    
    # Extract Whisper-specific configurations
    whisper_config = config.get('whisper', {})
    model_name = whisper_config.get('model_name', 'base')
    lang = whisper_config.get('lang', 'en')
    audio_max_length = whisper_config.get('audio_max_length', 480000)  # 30 seconds
    
    # Initialize Whisper tokenizer
    multilingual = True if 'large' in model_name or 'en' not in model_name else False
    tokenizer = whisper.tokenizer.get_tokenizer(
        multilingual=multilingual, 
        task='transcribe'
    )
    
    
    tr_dataset = Dataset(
        config['train'],
        tokenizer,
        model_name,
        max_length=audio_max_length,
        lang=lang
    )
    
    dv_dataset = Dataset(
        config['valid'],
        tokenizer,
        model_name,
        max_length=audio_max_length,
        lang=lang
    )
    
    batch_size = config.get('batch_size', 16)
    
    tr_loader = DataLoader(
        tr_dataset,
        batch_size=batch_size,
        collate_fn=WhisperDataCollatorWhithPadding,
        num_workers=n_jobs,
        pin_memory=use_gpu
    )
    
    dv_loader = DataLoader(
        dv_dataset,
        batch_size=batch_size,
        collate_fn=WhisperDataCollatorWhithPadding,
        num_workers=n_jobs,
        pin_memory=pin_memory
    )
    
    n_mels = 128 if 'large-v3' in model_name else 80
    feat_dim = n_mels
    
    data_msg = [
        f'Data spec. | Corpus = Whisper-{model_name} (lang: {lang})',
        f'           | Train sets = {len(tr_dataset)} utterances',
        f'           | Dev sets = {len(dv_dataset)} utterances',
        f'           | Batch size = {batch_size}',
        f'           | Audio max length = {audio_max_length} samples',
        f'I/O spec.  | Feature dim = {feat_dim} (Mel bands)',
    ]
    
    return tr_loader, dv_loader, feat_dim, tokenizer, data_msg