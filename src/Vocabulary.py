import torch
import typing
import os
from src.Data import Data

class Vocabulary:
    def __init__(self, data: Data | None = None) -> None:
        self.VOCABULARY_FILENAME = os.path.join('dataset', 'vocabulary.data')
        self.DELIMITER = '±'

        self.word_to_idx: dict[str, int] = {}
        self.idx_to_word: dict[int, str] = {}

        self.out_of_vocabulary_idx = -1
        self.out_of_vocabulary_token = '<oov>'

        self.padding_idx = -1
        self.padding_token = '<pad>'

        self.size = 0

        self.end_of_sequence_token = ';'

        if data:
            self._build_vocabulary_from_data(data)

    def __len__(self) -> int:
        return self.size
    
    def __getitem__(self, word: str) -> int:
        if word in self.word_to_idx:
            return self.word_to_idx[word]

        return self.out_of_vocabulary_idx
    
    def get_word(self, word_idx: int) -> str:
        if word_idx in self.idx_to_word:
            return self.idx_to_word[word_idx]
        
        return self.out_of_vocabulary_token

    def save(self) -> None:
        os.remove(self.VOCABULARY_FILENAME)
        with open(self.VOCABULARY_FILENAME, 'w+') as file:
            for word in self.word_to_idx:
                file.write(f'{word}{self.DELIMITER}{self.word_to_idx[word]}\n')

    def load(self) -> None:
        with open(self.VOCABULARY_FILENAME, 'r') as file:
            self._build_vocabulary_from_file(file)

    def vectorize(self, words: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        vector = torch.tensor([self[word] for word in words], dtype=torch.long)
        length = torch.tensor([len(words)], dtype=torch.long)

        return vector, length

    def _build_vocabulary_from_data(self, data: Data) -> None:
        word_idx = 0

        for input, target in data:
            for word in input:
                if word not in self.word_to_idx:
                    self.word_to_idx[word] = word_idx
                    self.idx_to_word[word_idx] = word
                    word_idx += 1
            
            if target not in self.word_to_idx:
                self.word_to_idx[target] = word_idx
                self.idx_to_word[word_idx] = target
                word_idx += 1

        self.out_of_vocabulary_idx = word_idx
        self.word_to_idx[self.out_of_vocabulary_token] = self.out_of_vocabulary_idx
        self.idx_to_word[self.out_of_vocabulary_idx] = self.out_of_vocabulary_token

        self.padding_idx = word_idx + 1
        self.word_to_idx[self.padding_token] = self.padding_idx
        self.idx_to_word[self.padding_idx] = self.padding_token

        self.size = word_idx + 2

    def _build_vocabulary_from_file(self, file: typing.TextIO) -> None:
        for line in file:
            word, word_idx_as_str = line.split(self.DELIMITER)
            word_idx = int(word_idx_as_str)

            self.word_to_idx[word] = word_idx
            self.idx_to_word[word_idx] = word

        self.out_of_vocabulary_idx = len(self.word_to_idx)
        self.word_to_idx[self.out_of_vocabulary_token] = self.out_of_vocabulary_idx
        self.idx_to_word[self.out_of_vocabulary_idx] = self.out_of_vocabulary_token

        self.padding_idx = self.out_of_vocabulary_idx + 1
        self.word_to_idx[self.padding_token] = self.padding_idx
        self.idx_to_word[self.padding_idx] = self.padding_token

        self.size = self.out_of_vocabulary_idx + 2
