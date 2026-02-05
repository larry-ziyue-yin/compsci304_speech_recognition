End-to-End ASR Assignment Instructions

This assignment is for project5 and project6, referencing the complete project [LHY/hw1/End-to-end-ASR-Pytorch-master](https://github.com/Alexander-H-Liu/End-to-end-ASR-Pytorch.git). You need to fill in the blanks at the specified locations and understand the core workflow, completing the training and testing of an end-to-end speech recognition system.

## Code you need to complete
### project5
- `src/asr.py`
  - **ASR.forward**
- `src/audio.py`
  - Rplace this file with the code you write in project1 to extract audio features.
- `src/data.py`
  - **collect_audio_batch**
  - `src/ctc.py`
  - **CTCBeamDecoder.forward**
- `bin/train_asr.py`
  - **fetch_data**
  - **set_model / loss**
  - **exec**

### project6
- `bin/tesr_asr.py`
  - **fetch_data**
  - **greedy_decode**

- `src/decode.py`
  - **fetch_data**
  - **BeamDecoder.forward**