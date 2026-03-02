import torch
import editdistance as ed
from src.solver import BaseSolver
from src.optim import Optimizer
from src.data import load_dataset_wtimit
from src.lora import inject_lora, get_trainable_params, trainable_param_count
from src.util import human_format, cal_er, feat_to_fig
import whisper

class Solver(BaseSolver):
    ''' Solver for training'''

    def __init__(self, config, paras, mode):
        super().__init__(config, paras, mode)
        # Logger settings
        self.best_wer = {'att': 3.0}

    def fetch_data(self, data):
        ''' Move data to device and compute text seq. length'''
        #===================TODO===============
        # 1. Get input_ids, labels, dec_input_ids
        # 2. Move input_ids, labels, dec_input_ids to self.device
        input_ids = data['input_ids'].to(self.device)
        labels = data['labels'].to(self.device)
        dec_input_ids = data['dec_input_ids'].to(self.device)

        #===================TODO===============
        return input_ids, labels, dec_input_ids


    def load_data(self):
        ''' Load data for training/validation, store tokenizer and input/output shape'''
        self.tr_set, self.dv_set, self.feat_dim, self.tokenizer, msg = \
            load_dataset_wtimit(self.paras.njobs, self.paras.gpu, self.paras.pin_memory,
                         self.config['data'], for_test=False)
        self.special_token_set = set(self.tokenizer.special_tokens.values())
        self.verbose(msg)

    def set_model(self):
        ''' Setup ASR model and optimizer '''
        whisper_cfg = self.config['data'].get('whisper', {})
        model_name = whisper_cfg.get('model_name', 'large')
        use_lora = bool(whisper_cfg.get('use_lora', False))
        lora_rank = int(whisper_cfg.get('lora_rank', 2))
        lora_alpha = float(whisper_cfg.get('lora_alpha', lora_rank))

        # Model
        #===================TODO===============
        # load Whisper model using whisper.load_model
        # input: self.config['data']['whisper']['model_name']
        # move to gpu
        self.model = whisper.load_model(model_name, device=self.device).to(self.device)
        if use_lora:
            replaced = inject_lora(
                self.model,
                rank=lora_rank,
                alpha=lora_alpha,
            )
            tr_n, all_n = trainable_param_count(self.model)
            self.verbose(
                f"LoRA enabled | rank={lora_rank} alpha={lora_alpha} | replaced={replaced} "
                f"| trainable={tr_n}/{all_n} ({100.0*tr_n/all_n:.4f}%)"
            )
        
        #===================TODO===============
        model_paras = [{'params': get_trainable_params(self.model)}]

        # Losses
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

        # Optimizer
        self.optimizer = Optimizer(model_paras, self.config['hparas']['optimizer'],self.config['hparas']['lr'],self.config['hparas']['eps'],self.config['hparas']['lr_scheduler'])
        self.verbose(self.optimizer.create_msg())

        # Enable AMP if needed
        self.enable_apex()

        # Automatically load pre-trained model if self.paras.load is given
        self.load_ckpt()

        # ToDo: other training methods

    def training_step(self, batch):
        ''' Training step for Whisper model '''
        input_ids, labels, dec_input_ids = batch
        
        # ==============================TODO ==============================:
        # 1. Use Whisper encoder(input_ids) to extract audio_features 
        # 2. Use decoder(dec_input_ids, audio_features) to obtain logits  
        # 3. Reshape logits and labels, then compute CrossEntropyLoss
        #
        # Notes:
        #   - Padding tokens in labels have already been set to -100
        #   - loss_fn has already been defined in set_model
        audio_features, _ = self.model.encoder(input_ids)
        out = self.model.decoder(dec_input_ids, audio_features)
        loss = self.loss_fn(out.reshape(-1, out.size(-1)), labels.reshape(-1))
        

        # ==============================TODO ==============================
        return loss, out, labels

    def validation_step(self, batch):
        ''' Validation step for Whisper model '''
        input_ids, labels, dec_input_ids = batch
        
        # Forward pass
        audio_features, _ = self.model.encoder(input_ids)
        out = self.model.decoder(dec_input_ids, audio_features)
        
        # Calculate loss
        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
        
        # Prepare tokens for evaluation
        labels[labels == -100] = self.tokenizer.eot
        tokens = torch.argmax(out, dim=2)
        
        # Set all decoder predictions after first eot to eot
        eot_find = (torch.where(tokens == self.tokenizer.eot, 1, 0))
        first_eot = torch.argmax(torch.arange(eot_find.shape[1], 0, -1).to(self.device) * eot_find, 
                                dim=1, keepdim=True)
        tokens[torch.arange(eot_find.shape[1]).to(self.device) > first_eot] = self.tokenizer.eot
        
        # Calculate accuracy
        mask = ~(tokens[:, 3:] == self.tokenizer.eot)
        n_correct = torch.sum(
            tokens[:, 3:].masked_select(mask).eq(labels[:, 3:].masked_select(mask))
        )
        total = torch.sum(mask)
        acc = n_correct.item() / (total.item() + 1e-8)
        acc = acc if acc < 1 else 0
        
        # Decode predictions and references
        o_list, l_list = [], []
        for o, l in zip(tokens, labels):
            o_list.append(
                self.tokenizer.decode(
                    [int(t.item()) for t in o if t.item() not in self.special_token_set]
                )
            )
            l_list.append(
                self.tokenizer.decode(
                    [int(t.item()) for t in l if t.item() not in self.special_token_set]
                )
            )
        
        # Calculate WER and CER (you need to implement wer_cer function or import)
        # For now, using placeholder
        wer, cer = self.calculate_wer_cer(o_list, l_list)

        
        return {
            "loss": loss,
            "cer": cer,
            "wer": wer,
            "acc": acc,
            "predictions": o_list,
            "references": l_list
        }

    def calculate_wer_cer(self, hypo, ref):
        ''' Calculate WER and CER '''
        total_word_err = 0
        total_word_count = 0
        total_char_err = 0
        total_char_count = 0

        for h, r in zip(hypo, ref):
            h_words = h.strip().split()
            r_words = r.strip().split()
            total_word_err += ed.eval(h_words, r_words)
            total_word_count += max(len(r_words), 1)

            h_chars = list(h.replace(" ", ""))
            r_chars = list(r.replace(" ", ""))
            total_char_err += ed.eval(h_chars, r_chars)
            total_char_count += max(len(r_chars), 1)

        wer = total_word_err / max(total_word_count, 1)
        cer = total_char_err / max(total_char_count, 1)
        return wer, cer

    def exec(self):
        ''' Training Whisper ASR system '''
        self.verbose(f'Total training steps {self.max_step}.')
        n_epochs = 0
        self.timer.set()

        while self.step < self.max_step:
            for batch in self.tr_set:
                # Pre-step: zero gradients
                self.optimizer.pre_step(self.step)
                
                # Fetch data
                input_ids, labels, dec_input_ids = self.fetch_data(batch)
                self.timer.cnt('rd')
                
                # Training step
                loss, out, labels_out = self.training_step((input_ids, labels, dec_input_ids))
                self.timer.cnt('fw')
                
                grad_norm = self.backward(loss)
                
                # Update step counter
                self.step += 1
                
                # Logging
                if (self.step == 1) or (self.step % self.PROGRESS_STEP == 0):
                    self.progress('Tr stat | Loss - {:.4f} | Grad. Norm - {:.2f} | {}'
                                    .format(loss.cpu().item(), grad_norm, self.timer.show()))
                    self.write_log('loss', {'tr': loss})
                
                # Validation
                if (self.step == 1) or (self.step % self.valid_step == 0):
                    self.validate()
                
                # Clean up
                torch.cuda.empty_cache()
                self.timer.set()
                
                if self.step > self.max_step:
                    break
            
            n_epochs += 1
        
        self.log.close()

    def validate(self):
        ''' Validation loop '''
        self.model.eval()
        dev_metrics = {'loss': [], 'cer': [], 'wer': [], 'acc': []}
        
        with torch.no_grad():
            for i, batch in enumerate(self.dv_set):
                self.progress('Valid step - {}/{}'.format(i + 1, len(self.dv_set)))
                
                # Fetch data
                input_ids, labels, dec_input_ids = self.fetch_data(batch)
                
                # Validation step
                results = self.validation_step((input_ids, labels, dec_input_ids))
                
                # Collect metrics
                dev_metrics['loss'].append(results['loss'].cpu().item())
                dev_metrics['cer'].append(results['cer'])
                dev_metrics['wer'].append(results['wer'])
                dev_metrics['acc'].append(results['acc'])
                
                # Show examples on tensorboard
                if i == len(self.dv_set) // 2:
                    for j in range(min(len(results['predictions']), self.DEV_N_EXAMPLE)):
                        self.write_log(f'pred_text_{j}', results['predictions'][j])
                        self.write_log(f'ref_text_{j}', results['references'][j])
        
        # Calculate average metrics
        avg_metrics = {k: sum(v) / len(v) for k, v in dev_metrics.items() if v}
        
        # Save checkpoint if performance improves
        if avg_metrics['wer'] < self.best_wer['att']:
            self.best_wer['att'] = avg_metrics['wer']
            self.save_checkpoint('best_whisper.pth', 'wer', avg_metrics['wer'])
        
        # Log metrics
        for metric_name, value in avg_metrics.items():
            self.write_log(metric_name, {'dv': value})
        
        self.save_checkpoint('latest.pth', 'wer', avg_metrics['wer'], show_msg=False)
        
        # Resume training mode
        self.model.train()
