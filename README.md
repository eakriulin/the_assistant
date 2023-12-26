# 🧑‍💻 The Assistant

An LSTM-based neural network for TypeScript code autocompletion.

## Setup

Download and enter the project:

```zsh
git clone https://github.com/eakriulin/the_assistant.git
cd the_assistant
```

Create and activate the virtual environment

```zsh
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies

```zsh
pip3 install -r requirements.txt
```

Download TypeScript language grammar

```zsh
git clone https://github.com/tree-sitter/tree-sitter-typescript.git
```

## Run

```zsh
usage: main.py [-h] [--train] [--test] [--demonstrate]

options:
  -h, --help     show this help message and exit
  --train        If passed, the neural network will be trained
  --test         If passed, the neural network will be evaluated on the test set
  --demonstrate  If passed, the neural network will be used to demonstrate some examples of code autocompletion
```

Example

```zsh
python3 main.py --train --test --demonstrate
```

## Modules

### TextPreprocessor

Implements pre-processing and tokenization of the given piece of code. Since the dataset is expected to be small, variable, function, method and class names are normalized so it's easier for the neural network to see the regularities. To simplify the processing, a given piece of code gets represented as a tree using an external library. Then, a set of regular expressions is applied to format and tokenize the code.

### Data

Reads the files provided in the dataset folder, extracts input sequences and their corresponding targets and stores them in memory.

### Vocabulary

Constructs the vocabulary from the given inputs and targets, stores it in memory and writes it to disk as a file. Later, the `.load` method can be used to construct the vocabulary directly from the file.

### Dataset

Extends the functionality of the `torch.utils.data.Dataset` class. Provides an iterator over the pairs of inputs and targets. Calls the vocabulary to vectorize inputs and targets before returning them.

### NeuralNetwork

Extends the functionality of the `torch.nn.Module` class. Consists of the Embedding layer, LSTM and Linear layer.

1. Embedding layer represents each token as an multi-dimensional vector.
2. LSTM processes the sequence of tokens one by one and encodes the information about the sequence in its hidden state.
3. Linear layer receives the hidden state from the LSTM and projects it onto the vector the number of dimensions of which is equal to the size of the vocabulary. Later, the information from this layer can be used to predict which token is more likely to appear after the considered sequence.

### NextTokenPredictor

Implements training, evaluation and prediction algorithms for the aforementioned neural network.

1. Training procedure consists of multiple epochs. Within the epoch, the neural network is trained on mini-batches. The main goal of this procedure is to minimize the cross-entropy loss function.
2. Evaluation procedure is done on a special validation set which the neural network has not seen. It outputs the perplexity and accuracy values.
3. Prediction algorithm implements the autocompletion procedure. Given a piece of code to autocomplete, it does the pre-processing and calls the neural network with the vectorized input. The softmax function is applied to convert the information from the last linear layer of the neural network into the probability distribution. For each token in the vocabulary, this distribution shows how likely it is that the aforementioned token follows the sequence. At the end, we sample from the distribution and enhance the original sequence with the picked token. The process continues until the end of sequence token is generated or the length of the generated sequence reaches a predefined threshold.

## Comments

Variable, function, method and class names are intended to be self-descriptive. The code is written in the way to be clear and understandable without additional comments.

> Every time you write a comment, you should grimace and feel the failure of your ability of expression.  
> Robert C. Martin (Author of Clean Code)

🧑‍💻
