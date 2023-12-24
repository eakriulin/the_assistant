import re

class TextPreprocessor:
    RE_SPACES_AND_LINEBREAKS = r'[\s\n\r]'
    RE_NAME = r'[a-zA-Z0-9_]+'
    RE_NUMBER = r'[0-9]+'
    RE_COMMENTS_RE = r'\s*\/\/.*$'

    @staticmethod
    def tokenize(text: str) -> list[str]:
        text = TextPreprocessor.clear_comments(text)

        text_length = len(text)
        tokens: list[str] = []
        current_word_chars: list[str] = []

        for i in range(0, text_length):
            char = text[i]
            if re.match(TextPreprocessor.RE_NAME, char):
                current_word_chars.append(char)
                continue

            next_char = text[i + 1] if i + 1 < text_length else None
            if char == '.' and next_char and re.match(TextPreprocessor.RE_NUMBER, next_char):
                current_word_chars.append(char)
                continue

            if len(current_word_chars) > 0:
                tokens.append("".join(current_word_chars))
                current_word_chars = []

            if re.match(TextPreprocessor.RE_SPACES_AND_LINEBREAKS, char):
                continue

            tokens.append(char)

        if len(current_word_chars) > 0:
            tokens.append("".join(current_word_chars))
            current_word_chars = []

        return tokens
    
    @staticmethod
    def clear_comments(text: str) -> str:
        return re.sub(TextPreprocessor.RE_COMMENTS_RE, '', text, flags=re.MULTILINE).strip()