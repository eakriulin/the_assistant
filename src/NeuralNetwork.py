import torch
from hyperparameters import HYPERPARAMETERS
from src.Vocabulary import Vocabulary

class NeuralNetwork(torch.nn.Module):
    def __init__(self, vocabulary: Vocabulary) -> None:
        super(NeuralNetwork, self).__init__()

        self.vocabulary = vocabulary

        self.embedding_size = HYPERPARAMETERS['EMBEDDING_SIZE']
        self.lstm_hidden_state_size = HYPERPARAMETERS['LSTM_HIDDEN_STATE_SIZE']
        
        self.embedding = torch.nn.Embedding(num_embeddings=vocabulary.size, embedding_dim=self.embedding_size, padding_idx=vocabulary.padding_idx)
        self.lstm = torch.nn.LSTM(input_size=self.embedding_size, hidden_size=self.lstm_hidden_state_size, batch_first=True)
        self.linear = torch.nn.Linear(in_features=self.lstm_hidden_state_size, out_features=vocabulary.size)

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        minibatch_size = len(inputs)

        embedded_inputs = self.embedding.forward(inputs) # note: (minibatch_size, sequence_length, embedding_size)
        encoded_inputs, _ = self.lstm.forward(embedded_inputs, self.initialize_hidden_and_cell_states(minibatch_size)) # note: (minibatch_size, sequence_length, lstm_hidden_state_size)

        indices_of_last_tokens = lengths - 1
        indices_of_last_tokens = indices_of_last_tokens.view(minibatch_size, 1, 1).repeat(1, 1, self.lstm_hidden_state_size)

        encoded_inputs = torch.gather(encoded_inputs, 1, indices_of_last_tokens) # note: (minibatch_size, 1, lstm_hidden_state_size)
        return self.linear.forward(encoded_inputs)

    def initialize_hidden_and_cell_states(self, minibatch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = torch.zeros(1, minibatch_size, self.lstm_hidden_state_size)
        cell_states = torch.zeros(1, minibatch_size, self.lstm_hidden_state_size)
        return (hidden_states, cell_states)
