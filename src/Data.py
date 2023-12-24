import os
from hyperparameters import HYPERPARAMETERS
from src.TextPreprocessor import TextPreprocessor

class Data:
    def __init__(self, folder: str | None = None, inputs: list[list[str]] | None = None, targets: list[str] | None = None) -> None:
        self.inputs: list[list[str]] = inputs if inputs else []
        self.targets: list[str] = targets if targets else []

        if folder:
            for filename in os.listdir(folder):
                self._extract_inputs_and_targets_from_file(folder, filename)

    def __len__(self) -> int:
        return len(self.inputs)
    
    def __getitem__(self, idx: int) -> tuple[list[str], str]:
        return self.inputs[idx], self.targets[idx]

    def _extract_inputs_and_targets_from_file(self, folder: str, filename: str) -> None:
        with open(os.path.join(folder, filename)) as file:
            text = file.read()
            tokens = TextPreprocessor.tokenize(text)

            number_of_tokens = len(tokens)
            last_token_idx = number_of_tokens - 1

            for i in range(0, number_of_tokens):
                target_idx = min(i + HYPERPARAMETERS["SEQUENCE_LENGTH"], last_token_idx)

                input = tokens[i:target_idx]
                target = tokens[target_idx]

                self.inputs.append(input)
                self.targets.append(target)

                if target_idx == last_token_idx:
                    break