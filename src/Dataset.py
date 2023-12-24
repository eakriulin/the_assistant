import torch
import torch.utils.data
from src.Data import Data
from src.Vocabulary import Vocabulary

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data: Data, vocabulary: Vocabulary) -> None:
        self.data = data
        self.vocabulary = vocabulary

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        input, target = self.data[idx]

        vectorized_input, vectorized_length = self.vocabulary.vectorize(input)
        vectorized_target, _ = self.vocabulary.vectorize([target])

        return vectorized_input, vectorized_length, vectorized_target