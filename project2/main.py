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
    parser.add_argument('--train_list', type=str, default='data/train.list',
                       help='Training list file')
    parser.add_argument('--test_list', type=str, default='data/test.list',
                       help='Testing list file')
    parser.add_argument('--features_dir', type=str, default='data/features',
                       help='Directory to save/load features')
    parser.add_argument('--n_states', type=int, default=5,
                       help='Number of HMM states')
    parser.add_argument('--max_iter', type=int, default=20,
                       help='Maximum iterations for training')
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
    
    

    print("\nPer-digit accuracy (GMM):")
    for d in range(10):
        total = np.sum(cm_gmm[d, :])
        correct = cm_gmm[d, d]
        acc = correct / total * 100 if total > 0 else 0
        print(f"  Digit {d}: {acc:.1f}% ({correct}/{total})")

if __name__ == "__main__":
    main()