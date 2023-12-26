import torch.utils.data
from src.NextTokenPredictor import NextTokenPredictor

def test(next_token_predictor: NextTokenPredictor, test_set: torch.utils.data.DataLoader) -> None:
    test_perplexity, test_accuracy = next_token_predictor.eval(test_set)
    print('\n------')
    print(f'\nTEST SET RESULTS:\nperplexity — {test_perplexity}\naccuracy — {test_accuracy}')

def demonstrate_examples(next_token_predictor: NextTokenPredictor) -> None:
    original_texts = [
        'export function _function_<T>(',
        'export class _class_<T = any> {',
        'const _variable0_ = new',
    ]

    print('\n------')
    for idx, original_text in enumerate(original_texts):
        autocompleted_text = next_token_predictor.predict(original_text)
        print(f'\nEXAMPLE {idx + 1}:\n🙋: {original_text}\n🤖: {autocompleted_text}')