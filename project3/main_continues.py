from src.grammar import DigitLoopGrammar
from src.composite_hmm import CompositeHMM
from src.continuous_decoder import viterbi_decode, states_to_digits
from src.wer import word_error_rate, edit_distance
from src.feature_extraction import FeatureExtractor
import numpy as np
from src.utils import load_hmm_models

penalties = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]  # Determine the optimal insertion penalty empirically

# Use the trained digit HMMs in project2. You can use "save_hmm_models(models_gmm, "models/hmm_gmm.pkl")" to save the model.
# digit_models = load_hmm_models("models/hmm_gmm.pkl")
# digit_models = load_hmm_models("models/hmm_gmm_zyh.pkl")
digit_models = load_hmm_models("models/hmm_gmm_nosil.pkl")

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
    total_edit_distance = 0
    total_ref_words = 0
    sent_correct = 0
    for audio, ref in test_data:
        feats = extractor.extract_features(audio)
        path = viterbi_decode(hmm, feats)
        hyp = states_to_digits(path, hmm)
        # print(hyp, ref)
        # Calculate edit distance at word level (each digit is a word)
        ref_list = list(ref)
        hyp_list = list(hyp)
        total_edit_distance += edit_distance(ref_list, hyp_list)
        total_ref_words += len(ref_list)
        sent_correct += (ref == hyp)

    print("Penalty:", p)
    print("Sentence accuracy:", sent_correct / len(test_data))
    print("WER:", total_edit_distance / total_ref_words if total_ref_words > 0 else 0.0)