# 🧑‍💻 The Assistant

A tiny neural network for TypeScript code autocompletion.

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
