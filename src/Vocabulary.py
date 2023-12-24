import torch
import typing
import os
from src.Data import Data

class Vocabulary:
    def __init__(self, data: Data | None = None) -> None:
        self.VOCABULARY_FILENAME = os.path.join('dataset', 'vocabulary.data')
        self.DELIMITER = '±'
        self.WORD_TO_MARSHALED_WORD = {
            '\n': '\\n'
        }
        self.MARSHALED_WORD_TO_WORD = {
            '\\n': '\n'
        }

        self.word_to_idx: dict[str, int] = {}
        self.idx_to_word: dict[int, str] = {}

        self.out_of_vocabulary_idx = -1
        self.out_of_vocabulary_token = '<oov>'

        self.padding_idx = -1
        self.padding_token = '<pad>'

        self.size = 0

        if data:
            self._build_vocabulary_from_data(data)

    def __len__(self) -> int:
        return self.size
    
    def __getitem__(self, word: str) -> int:
        if word in self.word_to_idx:
            return self.word_to_idx[word]

        return self.out_of_vocabulary_idx

    def save(self) -> None:
        with open(self.VOCABULARY_FILENAME, 'w+') as file:
            for word in self.word_to_idx:
                file.write(f'{self._marshal_word(word)}{self.DELIMITER}{self.word_to_idx[word]}\n')

    def load(self) -> None:
        with open(self.VOCABULARY_FILENAME, 'r') as file:
            self._build_vocabulary_from_file(file)

    def vectorize(self, words: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        vector = torch.tensor([self[word] for word in words], dtype=torch.long)
        length = torch.tensor([len(words)], dtype=torch.long)

        return vector, length
    
    def get_word(self, idx: int) -> str:
        return self.idx_to_word[idx]

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
            marshaled_word, word_idx = line.split(self.DELIMITER)

            word = self._unmarshal_word(marshaled_word)
            self.word_to_idx[word] = int(word_idx)
            self.idx_to_word[int(word_idx)] = word

        self.out_of_vocabulary_idx = len(self.word_to_idx)
        self.word_to_idx[self.out_of_vocabulary_token] = self.out_of_vocabulary_idx
        self.idx_to_word[self.out_of_vocabulary_idx] = self.out_of_vocabulary_token

        self.padding_idx = self.out_of_vocabulary_idx + 1
        self.word_to_idx[self.padding_token] = self.padding_idx
        self.idx_to_word[self.padding_idx] = self.padding_token

        self.size = self.out_of_vocabulary_idx + 2

    def _marshal_word(self, word: str) -> str:
        if word in self.WORD_TO_MARSHALED_WORD:
            return self.WORD_TO_MARSHALED_WORD[word]

        return word

    def _unmarshal_word(self, word: str) -> str:
        if word in self.MARSHALED_WORD_TO_WORD:
            return self.MARSHALED_WORD_TO_WORD[word]

        return word
