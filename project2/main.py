import os
import numpy as np
import argparse
from src.feature_extraction import FeatureExtractor
from src.kmeans_trainer import SegmentalKMeansTrainer
from src.utils import plot_confusion_matrix, save_results

def load_file_lists(train_list_file, test_list_file):
    train_dict = {str(d): [] for d in range(10)}
    test_dict = {str(d): [] for d in range(10)}
    
    with open(train_list_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                label, filepath = line.split(' ', 1)
                train_dict[label].append(filepath)
    
    with open(test_list_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                label, filepath = line.split(' ', 1)
                test_dict[label].append(filepath)
    
    return train_dict, test_dict

def save_features(features_list, filepath):
    np.save(filepath, np.array(features_list, dtype=object), allow_pickle=True)

def load_features(filepath):
    return np.load(filepath, allow_pickle=True).tolist()

def main():
    parser = argparse.ArgumentParser(description='HMM-based Isolated Digit Recognition')
    parser.add_argument('--train_list', type=str, default='train.list',
                       help='Training list file')
    parser.add_argument('--test_list', type=str, default='test.list',
                       help='Testing list file')
    parser.add_argument('--features_dir', type=str, default='data/features',
                       help='Directory to save/load features')
    parser.add_argument('--n_states', type=int, default=5,
                       help='Number of HMM states')
    parser.add_argument('--max_iter', type=int, default=20,
                       help='Maximum iterations for training')
    parser.add_argument('--plot_cm', action='store_true',
                       help='Plot confusion matrices after evaluation')
    parser.add_argument('--save_results', action='store_true',
                       help='Save evaluation results to JSON')
    parser.add_argument('--results_file', type=str, default='results.json',
                       help='Results JSON output path')
    args = parser.parse_args()
    
    os.makedirs(args.features_dir, exist_ok=True)
    
    print("Loading file lists...")
    train_dict, test_dict = load_file_lists(args.train_list, args.test_list)
    
    print("Extracting features...")
    extractor = FeatureExtractor()
    
    feature_files_exist = all(
        os.path.exists(os.path.join(args.features_dir, f"digit_{d}_train.npy")) and
        os.path.exists(os.path.join(args.features_dir, f"digit_{d}_test.npy"))
        for d in range(10)
    )
    
    if feature_files_exist:
        print("Loading pre-computed features...")
        train_data = {}
        test_data = {}
        for d in range(10):
            train_data[str(d)] = load_features(os.path.join(args.features_dir, f"digit_{d}_train.npy"))
            test_data[str(d)] = load_features(os.path.join(args.features_dir, f"digit_{d}_test.npy"))
    else:
        print("Extracting features from audio files...")
        train_data = {}
        test_data = {}
        
        for digit in range(10):
            train_data[str(digit)] = []
            for filepath in train_dict[str(digit)]:
                features = extractor.extract_features(filepath)
                if features is not None:
                    train_data[str(digit)].append(features)
            save_features(train_data[str(digit)], 
                         os.path.join(args.features_dir, f"digit_{digit}_train.npy"))
            print(f"  Digit {digit}: {len(train_data[str(digit)])} training samples")
        
        for digit in range(10):
            test_data[str(digit)] = []
            for filepath in test_dict[str(digit)]:
                features = extractor.extract_features(filepath)
                if features is not None:
                    test_data[str(digit)].append(features)
            save_features(test_data[str(digit)], 
                         os.path.join(args.features_dir, f"digit_{digit}_test.npy"))
            print(f"  Digit {digit}: {len(test_data[str(digit)])} testing samples")

    print("\nTraining samples per digit:")
    for d in range(10):
        print(f"  Digit {d}: {len(train_data[str(d)])}")
    
    print("\nTesting samples per digit:")
    for d in range(10):
        print(f"  Digit {d}: {len(test_data[str(d)])}")
    
    # Single Gaussian HMM
    print("Problem 1: Single Gaussian HMM")
    
    trainer_single = SegmentalKMeansTrainer(
        n_states=args.n_states,
        n_mixtures=1,
        max_iter=args.max_iter
    )
    
    # train
    print("Training single Gaussian HMMs...")
    models_single = trainer_single.train(train_data)
    
    # test
    print("Testing single Gaussian HMMs...")
    accuracy_single, cm_single = trainer_single.evaluate(test_data)
    print(f"Recognition Accuracy (Single Gaussian): {accuracy_single:.2f}%")
    if args.plot_cm:
        plot_confusion_matrix(cm_single, title="Confusion Matrix (Single Gaussian)")
    
    
    # GMM HMM (4 mixtures)
    print("Problem 2: GMM HMM (4 mixtures)")
    
    trainer_gmm = SegmentalKMeansTrainer(
        n_states=args.n_states,
        n_mixtures=4,
        max_iter=args.max_iter
    )
    
    # train
    print("Training GMM HMMs...")
    models_gmm = trainer_gmm.train(train_data)
    
    # test
    print("Testing GMM HMMs...")
    accuracy_gmm, cm_gmm = trainer_gmm.evaluate(test_data)
    print(f"Recognition Accuracy (GMM with 4 mixtures): {accuracy_gmm:.2f}%")
    if args.plot_cm:
        plot_confusion_matrix(cm_gmm, title="Confusion Matrix (GMM, 4 Mixtures)")
    
    

    def per_digit_accuracy(confusion_matrix):
        per_digit = {}
        for d in range(10):
            total = np.sum(confusion_matrix[d, :])
            correct = confusion_matrix[d, d]
            acc = correct / total * 100 if total > 0 else 0
            per_digit[str(d)] = {
                "accuracy": acc,
                "correct": int(correct),
                "total": int(total),
            }
        return per_digit

    print("\nPer-digit accuracy (GMM):")
    per_digit_gmm = per_digit_accuracy(cm_gmm)
    for d in range(10):
        acc = per_digit_gmm[str(d)]["accuracy"]
        correct = per_digit_gmm[str(d)]["correct"]
        total = per_digit_gmm[str(d)]["total"]
        print(f"  Digit {d}: {acc:.1f}% ({correct}/{total})")

    if args.save_results:
        results = {
            "settings": {
                "n_states": args.n_states,
                "max_iter": args.max_iter,
            },
            "single_gaussian": {
                "n_mixtures": 1,
                "accuracy": accuracy_single,
                "confusion_matrix": cm_single.tolist(),
                "per_digit": per_digit_accuracy(cm_single),
            },
            "gmm": {
                "n_mixtures": 4,
                "accuracy": accuracy_gmm,
                "confusion_matrix": cm_gmm.tolist(),
                "per_digit": per_digit_gmm,
            },
        }
        save_results(results, filename=args.results_file)

if __name__ == "__main__":
    main()