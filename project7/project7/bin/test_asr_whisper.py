import os
import torch
import editdistance as ed
from tqdm import tqdm
import whisper
from whisper.tokenizer import get_tokenizer

from src.solver import BaseSolver
from src.data import load_dataset_wtimit
from src.lora import inject_lora

class Solver(BaseSolver):
    ''' Solver for Whisper testing'''

    def __init__(self, config, paras, mode):
        super().__init__(config, paras, mode)


        # Whisper specific settings
        self.model_name = config['data']['whisper']['model_name']
        self.lang = config['data']['whisper']['lang']
        self.multilingual = True if 'large' in self.model_name or 'en' not in self.model_name else False
        self.tokenizer = get_tokenizer(multilingual=self.multilingual, task='transcribe')
        self.special_token_set = set(self.tokenizer.special_tokens.values())

        # Output file
        self.output_file = str(self.ckpdir) + '_{}_{}.csv'

        # Override batch size for beam decoding
        self.greedy = self.config['decode']['beam_size'] == 1
        if not self.greedy:
            self.config['data']['batch_size'] = 1
        
        self.step = 0
    
    def fetch_data(self, data):
        ''' Move data to device for Whisper model '''
        # Assuming data format: (name, mel_spec, labels, dec_input_ids)
        # or adapt to your dataset format
        #==================TODO===================
        name, input_ids, labels, dec_input_ids = data['name'], data['input_ids'], data['labels'], data['dec_input_ids']
        # move input_ids, labels, dec_input_ids to device
        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)
        dec_input_ids = dec_input_ids.to(self.device)

        #==================TODO===================
        return name, input_ids, labels, dec_input_ids 


    def load_data(self):
        ''' Load data for testing '''
        self.dv_set, self.tt_set, self.feat_dim, self.tokenizer, msg = \
            load_dataset_wtimit(self.paras.njobs, self.paras.gpu, self.paras.pin_memory,
                         self.config['data'], for_test=True)
        self.special_token_set = set(self.tokenizer.special_tokens.values())
        self.verbose(msg)

    def set_model(self):
        ''' Setup Whisper model for testing '''
        whisper_cfg = self.config['data'].get('whisper', {})
        local_model_path = whisper_cfg.get('local_model_path', '')
        use_lora = bool(whisper_cfg.get('use_lora', False))
        lora_rank = int(whisper_cfg.get('lora_rank', 2))
        lora_alpha = float(whisper_cfg.get('lora_alpha', lora_rank))
        
        #==================TODO===================
        # Load Whisper model
        model_ref = self.model_name
        if local_model_path:
            model_ref = os.path.expanduser(local_model_path)
            if not os.path.isabs(model_ref):
                model_ref = os.path.join(os.getcwd(), model_ref)
            if not os.path.isfile(model_ref):
                raise FileNotFoundError(
                    f"local_model_path not found: {model_ref}"
                )
            self.verbose(f"Loading Whisper from local checkpoint: {model_ref}")
        self.model = whisper.load_model(model_ref, device=self.device).to(self.device)
        if use_lora:
            replaced = inject_lora(self.model, rank=lora_rank, alpha=lora_alpha)
            self.verbose(
                f"LoRA enabled for decoding | rank={lora_rank} alpha={lora_alpha} | replaced={replaced}"
            )

        #==================TODO===================
        self.verbose(f"Loaded Whisper model: {self.model_name}")
        
        # Load checkpoint if specified
        if self.paras.load and os.path.isfile(self.paras.load):
            self.load_ckpt()
            self.verbose(f"Loaded checkpoint from {self.paras.load}")
        elif self.paras.load:
            self.verbose(f"Checkpoint not found ({self.paras.load}), use base Whisper weights.")
        
        # Set to evaluation mode
        self.model.eval()
        
        # Greedy decoding (Whisper uses greedy by default)
        self.greedy = self.config['decode']['beam_size'] == 1
        
        # For beam decoding (optional - implement if needed)
        if not self.greedy and 'beam_size' in self.config['decode']:
            self.verbose(f"Beam search with size {self.config['decode']['beam_size']}")
        
        self.verbose("Model setup complete")

    def whisper_greedy_decode(self, feat):
        ''' Greedy decoding for Whisper '''
        with torch.no_grad():
            # Forward through encoder
            audio_features, _ = self.model.encoder(feat)
            
            # Initialize decoder input with special tokens
            # For multilingual: [sot, lang_token, transcribe_token, no_timestamps]
            initial_tokens = [
                self.tokenizer.sot,
                self.tokenizer.special_tokens.get(f"<|{self.lang}|>", self.tokenizer.sot),
                self.tokenizer.transcribe,
                self.tokenizer.no_timestamps
            ]
            
            # Prepare decoder input
            dec_input = torch.tensor([initial_tokens], device=self.device)
            
            # Greedy decoding loop
            max_len = 448  # Whisper max tokens
            for i in range(max_len):
                # Forward decoder
                logits = self.model.decoder(dec_input, audio_features)
                
                # Get next token
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                
                # Append to decoder input
                dec_input = torch.cat([dec_input, next_token], dim=-1)
                
                # Check for EOT
                if next_token.item() == self.tokenizer.eot:
                    break
            
            return dec_input[0].cpu().tolist()

    def whisper_transcribe(self, feat, language=None):
        ''' Use Whisper's built-in decode function '''
        with torch.no_grad():
            # Create options for decoding
            options = whisper.DecodingOptions(
                language=language or self.lang,
                without_timestamps=True,
                fp16=(self.device.type == 'cuda'),
                beam_size=None if self.greedy else self.config['decode'].get('beam_size', 20),
                task='transcribe'
            )
            
            # Decode 
            return whisper.decode(self.model, feat, options)

    def greedy_decode(self, dv_set, use_whisper_api=True):
        ''' Greedy Decoding for Whisper '''
        results = []
        
        for i, data in enumerate(dv_set):
            self.progress(f'Decoding step - {i+1}/{len(dv_set)}')
            
            # Fetch data
            name, feat, labels, dec_input_ids = self.fetch_data(data)
            
            if use_whisper_api:
                # Batch processing: 
                decode_results = self.whisper_transcribe(feat)
                
                # decode_results 
                for j, result in enumerate(decode_results):
                    # 获取识别结果
                    hyp_text = result.text.strip()
                    
                    # Get ground truth if available
                    if labels is not None:
                        # Decode labels to text (remove special tokens)
                        label_seq = labels[j]
                        # Filter out special tokens and padding
                        valid_tokens = [
                            int(t.item())
                            for t in label_seq
                            if t.item() not in self.special_token_set and t.item() != -100
                        ]
                        true_text = self.tokenizer.decode(valid_tokens)
                    else:
                        true_text = ""
                    
                    idx = name[j] if name is not None else f"{i}_{j}"
                    results.append((str(idx), hyp_text, true_text))
                    
                    # print examples
                    # if i == 0 and j < 2:  
                    #     self.verbose(f"Batch {i}, Sample {j}:")
                    #     self.verbose(f"  HYP: {hyp_text}")
                    #     self.verbose(f"  REF: {true_text}")
            else:
                # Custom greedy decoding
                for j in range(feat.size(0)):
                    single_feat = feat[j:j+1]
                    hyp_tokens = self.whisper_greedy_decode(single_feat)
                    
                    # Remove special tokens from hypothesis
                    hyp_tokens_clean = [t for t in hyp_tokens if t not in self.special_token_set]
                    hyp_text = self.tokenizer.decode(hyp_tokens_clean).strip()
                    
                    # Get ground truth if available
                    if labels is not None:
                        label_seq = labels[j]
                        valid_tokens = [
                            int(t.item())
                            for t in label_seq
                            if t.item() not in self.special_token_set and t.item() != -100
                        ]
                        true_text = self.tokenizer.decode(valid_tokens)
                    else:
                        true_text = ""
                    
                    idx = name[j] if name is not None else f"{i}_{j}"
                    results.append((str(idx), hyp_text, true_text))
        
        return results

    def exec(self):
        ''' Testing Whisper ASR system '''
        self.verbose("Starting Whisper testing...")
        
        # Test on both dev and test sets
        for s, ds in zip(['dev', 'test'], [self.dv_set, self.tt_set]):
            if ds is None:
                continue
                
            # Setup output file
            self.cur_output_path = self.output_file.format(s, 'output')
            with open(self.cur_output_path, 'w', encoding='UTF-8') as f:
                f.write('idx\thyp\ttruth\n')
            
            # Greedy decode (Whisper's default)
            self.verbose(f'Performing greedy decoding on {s} set, num of batch = {len(ds)}.')
            
            # Use Whisper's built-in API for better results
            use_whisper_api = True
            results = self.greedy_decode(ds, use_whisper_api=use_whisper_api)
            
            self.verbose(f'Results will be stored at {self.cur_output_path}')
            
            # Write results
            self.write_hyp(results, self.cur_output_path)
            
            # Calculate WER/CER if ground truth is available
            if any(true_text for _, _, true_text in results):
                self.calculate_metrics(results, s)
        
        self.verbose('All testing done!')

    def write_hyp(self, results, output_path):
        ''' Write decoding results to file '''
        for idx, (name, hyp, truth) in enumerate(tqdm(results, desc="Writing results")):
            # Clean up text
            hyp_clean = hyp.strip()
            truth_clean = truth.strip()
            
            with open(output_path, 'a', encoding='UTF-8') as f:
                f.write(f'{name}\t{hyp_clean}\t{truth_clean}\n')
            
            # Print some examples
            if idx < 3:
                self.verbose(f"Example {name}:")
                self.verbose(f"  HYP: {hyp_clean}")
                self.verbose(f"  REF: {truth_clean}")
                self.verbose("-" * 50)

    def calculate_metrics(self, results, dataset_name):
        ''' Calculate WER and CER metrics '''
        hypotheses = []
        references = []

        for _, hyp, ref in results:
            if ref:
                hypotheses.append(hyp)
                references.append(ref)

        if not references:
            self.verbose(f"No ground truth available for {dataset_name} set")
            return

        total_word_err = 0
        total_word_count = 0
        total_char_err = 0
        total_char_count = 0

        for hyp, ref in zip(hypotheses, references):
            hyp_words = hyp.strip().split()
            ref_words = ref.strip().split()
            total_word_err += ed.eval(hyp_words, ref_words)
            total_word_count += max(len(ref_words), 1)

            hyp_chars = list(hyp.replace(" ", ""))
            ref_chars = list(ref.replace(" ", ""))
            total_char_err += ed.eval(hyp_chars, ref_chars)
            total_char_count += max(len(ref_chars), 1)

        wer_score = total_word_err / max(total_word_count, 1)
        cer_score = total_char_err / max(total_char_count, 1)

        self.verbose(f"{dataset_name} set metrics:")
        self.verbose(f"  WER: {wer_score:.4f} ({wer_score*100:.2f}%)")
        self.verbose(f"  CER: {cer_score:.4f} ({cer_score*100:.2f}%)")
