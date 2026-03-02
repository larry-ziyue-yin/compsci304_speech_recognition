import torch
from torch.utils.data import DataLoader
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
    input_ids = [
        np.pad(
            np.asarray(audio),
            ((0, 0), (0, max_audio_len - audio_len)),
            mode="constant",
            constant_values=0.0,
        )
        for audio, audio_len in zip(input_ids, audio_lengths)
    ]

    label_lengths = [len(lab) for lab in labels]
    dec_input_ids_length = [len(e) for e in dec_input_ids]
    max_label_len = max(label_lengths+dec_input_ids_length) # seems redundant

    #==============================TODO===============================
    # pad labels and dec_input_ids (np.pad)
    # pad the labels with -100 (dummy, ignore index in cross-entropy), pad the dec_input_ids with eot 50257
    labels = [
        np.pad(
            np.asarray(lab, dtype=np.int64),
            (0, max_label_len - len(lab)),
            mode="constant",
            constant_values=-100,
        )
        for lab in labels
    ]
    dec_input_ids = [
        np.pad(
            np.asarray(dec, dtype=np.int64),
            (0, max_label_len - len(dec)),
            mode="constant",
            constant_values=50257,
        )
        for dec in dec_input_ids
    ]

    #==============================TODO===============================
    batch = {
        "input_ids": input_ids,
        "labels": labels,
        "dec_input_ids": dec_input_ids,
    }

    batch = {k: torch.tensor(np.array(v), requires_grad=False) for k, v in batch.items()}

    batch["name"] = names
    return batch
    

def load_dataset_wtimit(n_jobs, use_gpu, pin_memory, config, for_test=False):
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
    
    valid_manifest = config.get('valid', config.get('dev'))
    if valid_manifest is None:
        raise KeyError("data.valid (or data.dev) is required in config")
    
    batch_size = config.get('batch_size', 16)

    dv_dataset = Dataset(
        valid_manifest,
        tokenizer,
        model_name,
        max_length=audio_max_length,
        lang=lang
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

    if for_test:
        test_manifest = config.get('test')
        if test_manifest is None:
            raise KeyError("data.test is required in test config")

        tt_dataset = Dataset(
            test_manifest,
            tokenizer,
            model_name,
            max_length=audio_max_length,
            lang=lang
        )
        tt_loader = DataLoader(
            tt_dataset,
            batch_size=batch_size,
            collate_fn=WhisperDataCollatorWhithPadding,
            num_workers=n_jobs,
            pin_memory=pin_memory
        )

        data_msg = [
            f'Data spec. | Corpus = Whisper-{model_name} (lang: {lang})',
            f'           | Dev sets = {len(dv_dataset)} utterances',
            f'           | Test sets = {len(tt_dataset)} utterances',
            f'           | Batch size = {batch_size}',
            f'           | Audio max length = {audio_max_length} samples',
            f'I/O spec.  | Feature dim = {feat_dim} (Mel bands)',
        ]
        return dv_loader, tt_loader, feat_dim, tokenizer, data_msg

    train_manifest = config.get('train')
    if train_manifest is None:
        raise KeyError("data.train is required in training config")

    tr_dataset = Dataset(
        train_manifest,
        tokenizer,
        model_name,
        max_length=audio_max_length,
        lang=lang
    )

    tr_loader = DataLoader(
        tr_dataset,
        batch_size=batch_size,
        collate_fn=WhisperDataCollatorWhithPadding,
        num_workers=n_jobs,
        pin_memory=pin_memory
    )

    data_msg = [
        f'Data spec. | Corpus = Whisper-{model_name} (lang: {lang})',
        f'           | Train sets = {len(tr_dataset)} utterances',
        f'           | Dev sets = {len(dv_dataset)} utterances',
        f'           | Batch size = {batch_size}',
        f'           | Audio max length = {audio_max_length} samples',
        f'I/O spec.  | Feature dim = {feat_dim} (Mel bands)',
    ]

    return tr_loader, dv_loader, feat_dim, tokenizer, data_msg
