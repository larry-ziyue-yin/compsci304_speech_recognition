# Lesson6: Fine-tuning Whisper on wTIMIT (Assignment Version)

## Dependencies
- `torch`, `torchaudio`, `openai-whisper`
- It is recommended to install the GPU version of torch to accelerate training; wTIMIT is small, so a small model (base) can be used to complete the task.

- `src/data.py`
  - **WhisperDataCollatorWhithPadding**
- `src/wtimit.py`
  - **__getitem__**
- `bin/test_asr_whisper.py`
  - **Solver.fetch_data**
  - **Solver.set_model**
- `bin/train_asr_whisper.py`
  - **Solver.fetch_data**
  - **Solver.set_model**
  - **Solver.training_step**