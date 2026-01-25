from src.grammar import DigitLoopGrammar
from src.composite_hmm import CompositeHMM
from src.continuous_decoder import viterbi_decode, states_to_digits
from src.wer import word_error_rate
from src.feature_extraction import FeatureExtractor
import numpy as np
from src.utils import load_hmm_models

penalties = [i * 0.1 for i in range(-10, 11)] # Determine the optimal insertion penalty empirically

# Use the trained digit HMMs in project2. You can use "save_hmm_models(models_gmm, "models/hmm_gmm.pkl")" to save the model.
# digit_models = load_hmm_models("models/hmm_gmm.pkl")
digit_models = load_hmm_models("models/hmm_gmm_zyh.pkl")

grammar = DigitLoopGrammar()
extractor = FeatureExtractor()
test_data=[]
with open('test.list','r') as f:
    for line in f:
        ref, path = line.strip().split(' ')
        test_data.append([path, ref])
for p in penalties:
    hmm = CompositeHMM()
    hmm.build(grammar, digit_models, p)
    total_wer = 0
    sent_correct = 0
    for audio, ref in test_data:
        feats = extractor.extract_features(audio)
        path = viterbi_decode(hmm, feats)
        hyp = states_to_digits(path, hmm)
        # print(hyp, ref)
        total_wer += word_error_rate(ref, hyp)
        sent_correct += (ref == hyp)

    print("Penalty:", p)
    print("Sentence accuracy:", sent_correct / len(test_data))
    print("WER:", total_wer / sum(len(r) for _, r in test_data))