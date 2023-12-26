import torch
import torch.utils.data
import os
import argparse
from hyperparameters import HYPERPARAMETERS
from src.Data import Data
from src.Vocabulary import Vocabulary
from src.Dataset import Dataset
from src.NeuralNetwork import NeuralNetwork
from src.NextTokenPredictor import NextTokenPredictor
from src.utils import test, demonstrate_examples

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='If passed, the neural network will be trained')
    parser.add_argument('--test', action='store_true', help='If passed, the neural network will be evaluated on the test set')
    parser.add_argument('--demonstrate', action='store_true', help='If passed, the neural network will be used to demonstrate some examples of code autocompletion')
    args = parser.parse_args()

    train_data = Data(folder=os.path.join('dataset', 'train'))
    val_data = Data(folder=os.path.join('dataset', 'val'))
    test_data = Data(folder=os.path.join('dataset', 'test'))

    vocabulary = Vocabulary(train_data)
    vocabulary.save()

    train_set = torch.utils.data.DataLoader(dataset=Dataset(train_data, vocabulary), batch_size=HYPERPARAMETERS['BATCH_SIZE'], shuffle=False, num_workers=1)
    val_set = torch.utils.data.DataLoader(dataset=Dataset(val_data, vocabulary), batch_size=HYPERPARAMETERS['BATCH_SIZE'], shuffle=False, num_workers=1)
    test_set = torch.utils.data.DataLoader(dataset=Dataset(test_data, vocabulary), batch_size=HYPERPARAMETERS['BATCH_SIZE'], shuffle=False, num_workers=1)

    neural_network = NeuralNetwork(vocabulary)
    next_token_predictor = NextTokenPredictor(neural_network)

    if args.train:
        next_token_predictor.train(train_set, val_set, HYPERPARAMETERS['NUMBER_OF_EPOCHS'], HYPERPARAMETERS['LEARNING_RATE'])

    next_token_predictor.load()

    if args.test:
        test(next_token_predictor, test_set)

    if args.demonstrate:
        demonstrate_examples(next_token_predictor)
