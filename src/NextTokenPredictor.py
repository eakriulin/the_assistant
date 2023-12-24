import torch
import torch.utils.data
import os
from src.NeuralNetwork import NeuralNetwork
from src.TextPreprocessor import TextPreprocessor

class NextTokenPredictor:
    def __init__(self, neural_network: NeuralNetwork) -> None:
        self.STATE_FILENAME = os.path.join('best_state.pth')
        self.MAX_AUTOCOMPLETIONS_COUNT = 16

        self.neural_network = neural_network
        self.end_of_sequence_token = ';'

    def save(self) -> None:
        torch.save(self.neural_network.state_dict(), self.STATE_FILENAME)

    def load(self) -> None:
        self.neural_network.load_state_dict(torch.load(self.STATE_FILENAME))

    def train(self, train_set: torch.utils.data.DataLoader, val_set: torch.utils.data.DataLoader, number_of_epochs: int, learning_rate: float) -> None:
        self.neural_network.train()
        optimizer = torch.optim.Adam(self.neural_network.parameters(), learning_rate)

        best_val_perplexity = float('inf')

        for e in range(0, number_of_epochs):
            epoch_loss = 0
            epoch_perplexity = 0
            epoch_accuracy = 0
            epoch_number_of_examples = 0

            for inputs, lengths, targets in train_set:
                optimizer.zero_grad()

                As = self.neural_network.forward(inputs, lengths)
                minibatch_loss = self._calculate_loss(As, targets)

                minibatch_loss.backward()
                optimizer.step()

                with torch.no_grad():
                    minibatch_number_of_examples = len(inputs)
                    minibatch_perplexity = self._calculate_perplexity(minibatch_loss)
                    minibatch_accuracy = self._calculate_accuracy(As, targets)

                    epoch_loss += minibatch_loss.item() * minibatch_number_of_examples
                    epoch_accuracy += minibatch_accuracy * minibatch_number_of_examples
                    epoch_number_of_examples += minibatch_number_of_examples

                    print(f'\tminibatch: train loss {minibatch_loss.item()}, train perplexity: {minibatch_perplexity}, train accuracy {minibatch_accuracy}')

            print('\nevaluating...')
            epoch_val_perplexity, epoch_val_accuracy = self.eval(val_set)
            if epoch_val_perplexity < best_val_perplexity:
                best_val_perplexity = epoch_val_perplexity
                self.save()

            epoch_loss /= epoch_number_of_examples
            epoch_perplexity = self._calculate_perplexity(torch.tensor(epoch_loss))
            epoch_accuracy /= epoch_number_of_examples

            print(f'\nepoch {e + 1} — train loss: {epoch_loss}, train perplexity: {epoch_perplexity}, train accuracy {epoch_accuracy}, val perplexity: {epoch_val_perplexity}, val accuracy {epoch_val_accuracy} | BEST val perplexity {best_val_perplexity}\n')

    def eval(self, val_set: torch.utils.data.DataLoader) -> float:
        with torch.no_grad():
            has_been_in_train_mode = self.neural_network.training
            self.neural_network.eval()

            number_of_examples = 0
            loss = 0
            perplexity = float('inf')
            accuracy = 0

            for inputs, lengths, targets in val_set:
                As = self.neural_network.forward(inputs, lengths)
                minibatch_loss = self._calculate_loss(As, targets)
                minibatch_accuracy = self._calculate_accuracy(As, targets)

                minibatch_number_of_examples = len(inputs)

                loss += minibatch_loss.item() * minibatch_number_of_examples
                accuracy += minibatch_accuracy * minibatch_number_of_examples
                number_of_examples += minibatch_number_of_examples

            loss /= number_of_examples
            perplexity = self._calculate_perplexity(torch.tensor(loss))
            accuracy /= number_of_examples

            if has_been_in_train_mode:
                self.neural_network.train()

        return perplexity, accuracy
    
    def predict(self, text: str) -> str:
        with torch.no_grad():
            self.neural_network.eval()

            tokens = TextPreprocessor.tokenize(text)

            next_token_idx = -1
            autocompletions_count = 0

            while next_token_idx != self.neural_network.vocabulary[self.end_of_sequence_token] and autocompletions_count < self.MAX_AUTOCOMPLETIONS_COUNT:
                input, length = self.neural_network.vocabulary.vectorize(tokens)

                input = input.reshape(1, input.shape[0])
                length = length.reshape(1, length.shape[0])

                predictions = self.neural_network.forward(input, length)
                next_token_idx = int(self._extract_predictions(predictions))
                next_token = self.neural_network.vocabulary.get_word(next_token_idx)

                tokens.append(next_token)
                autocompletions_count += 1

            return " ".join(tokens)

    def _calculate_loss(self, As: torch.Tensor, Ys: torch.Tensor) -> torch.Tensor:
        return torch.nn.CrossEntropyLoss().forward(As.squeeze(), Ys.squeeze())
    
    def _calculate_perplexity(self, loss: torch.Tensor):
        return torch.exp(loss).item()
    
    def _calculate_accuracy(self, As: torch.Tensor, Ys: torch.Tensor):
        correct_predictions = (self._extract_predictions(As) == Ys).to(dtype=torch.float32)
        return torch.mean(correct_predictions).item() * 100
    
    def _extract_predictions(self, As: torch.Tensor) -> torch.Tensor:
        probability_distributions = torch.nn.functional.softmax(As, dim=2)
        return torch.argmax(probability_distributions, dim=2)
    