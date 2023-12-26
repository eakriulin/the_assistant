import re
from tree_sitter import Language, Parser, Node, Tree

class TextPreprocessor:
    RE_STRING_TAGS = r'[\'\"\`]+'
    RE_NAME = r'[a-zA-Z0-9_]+'
    RE_NUMBER = r'[0-9]+'
    RE_SPACES_AND_LINEBREAKS = r'[\s\n\r]'

    RE_COMMENTS = r'\s*\/\/.*$'
    RE_IMPORTS = r'^import.+$'
    RE_CLASS = r'(class\s)'
    RE_FUNCTION = r'(function\s)(.+\s)?'
    RE_METHOD = r'(public\s|private\s)(.+\s)?'
    RE_NEGATE_ALPHANUMERIC_AND_DOT = r'(?<![\w\.])'
    RE_WORD_BOUNDARY = r'\b'

    @staticmethod
    def preprocess(text: str) -> str:
        text = TextPreprocessor.clear_comments(text)
        text = TextPreprocessor.clear_imports(text)
        text = TextPreprocessor.normalize_names(text)
        return text

    @staticmethod
    def tokenize(text: str) -> list[str]:
        text = TextPreprocessor.preprocess(text)

        text_length = len(text)
        tokens: list[str] = []

        is_processing_string = False
        current_string_chars: list[str] = []

        current_name_chars: list[str] = []

        for i in range(0, text_length):
            char = text[i]
            if re.match(TextPreprocessor.RE_STRING_TAGS, char):
                is_processing_string = not is_processing_string
                current_string_chars.append(char)
                continue

            if is_processing_string:
                current_string_chars.append(char)
                continue

            if re.match(TextPreprocessor.RE_NAME, char):
                current_name_chars.append(char)
                continue

            next_char = text[i + 1] if i + 1 < text_length else None
            if char == '.' and next_char and re.match(TextPreprocessor.RE_NUMBER, next_char):
                current_name_chars.append(char)
                continue

            if len(current_string_chars) > 0:
                tokens.append("".join(current_string_chars))
                current_string_chars = []

            if len(current_name_chars) > 0:
                tokens.append("".join(current_name_chars))
                current_name_chars = []

            if re.match(TextPreprocessor.RE_SPACES_AND_LINEBREAKS, char):
                continue

            tokens.append(char)

        if len(current_string_chars) > 0:
            tokens.append("".join(current_string_chars))
            current_string_chars = []

        if len(current_name_chars) > 0:
            tokens.append("".join(current_name_chars))
            current_name_chars = []

        return tokens
    
    @staticmethod
    def clear_comments(text: str) -> str:
        return re.sub(TextPreprocessor.RE_COMMENTS, '', text, flags=re.MULTILINE).strip()

    @staticmethod
    def clear_imports(text: str) -> str:
        return re.sub(TextPreprocessor.RE_IMPORTS, '', text, flags=re.MULTILINE).strip()
    
    @staticmethod
    def normalize_names(text: str) -> str:
        text = TextPreprocessor._normalize_names_in_classes(text)
        text = TextPreprocessor._normalize_names_in_functions_and_methods(text)
        return text
    
    @staticmethod
    def _normalize_names_in_classes(text: str) -> str:
        tree = TextPreprocessor._extract_code_tree(text)
        replacements: list[tuple[str, int, int]] = []
        stack: list[Node] = [tree.root_node]

        while len(stack) > 0:
            node = stack.pop()

            if node.type == 'class_declaration':
                class_text = TextPreprocessor._replace_names_in_class(node)
                replacements.append((class_text, node.start_byte, node.end_byte))

            for child in reversed(node.children):
                stack.append(child)

        # note: applying replacements from the end to the beginning to avoid offset issues
        for replacement, startIdx, endIdx in reversed(replacements):
            text = text[:startIdx] + replacement + text[endIdx:]

        return text
    
    @staticmethod
    def _normalize_names_in_functions_and_methods(text: str) -> str:
        tree = TextPreprocessor._extract_code_tree(text)
        replacements: list[tuple[str, int, int]] = []
        stack: list[Node] = [tree.root_node]

        while len(stack) > 0:
            node = stack.pop()

            if node.type in ['function_declaration', 'function_signature']:
                function_text = TextPreprocessor._replace_names_in_functions_and_methods(node)
                replacements.append((function_text, node.start_byte, node.end_byte))

            if node.type in ['method_definition', 'method_signature']:
                method_text = TextPreprocessor._replace_names_in_functions_and_methods(node)
                replacements.append((method_text, node.start_byte, node.end_byte))

            for child in reversed(node.children):
                stack.append(child)

        # note: applying replacements from the end to the beginning to avoid offset issues
        for replacement, startIdx, endIdx in reversed(replacements):
            text = text[:startIdx] + replacement + text[endIdx:]

        return text
    
    @staticmethod
    def _replace_names_in_functions_and_methods(node: Node) -> str:
        function_names: list[str] = []
        method_names: list[str] = []
        variable_names: list[str] = []

        root = node
        stack: list[Node] = [root]

        while len(stack) > 0:
            node = stack.pop()

            if TextPreprocessor._is_function_name(node):
                function_name = node.text.decode()
                function_names.append(function_name)

            if TextPreprocessor._is_method_name(node):
                method_name = node.text.decode()
                method_names.append(method_name)

            if TextPreprocessor._is_variable_name(node):
                variable_name = node.text.decode()
                variable_names.append(variable_name)

            for child in reversed(node.children):
                stack.append(child)
        
        function_name_to_replacement = TextPreprocessor._map_name_to_replacement(function_names, 'function')
        method_name_to_replacement = TextPreprocessor._map_name_to_replacement(method_names, 'method')
        variable_name_to_replacement = TextPreprocessor._map_name_to_replacement_with_idx(variable_names, 'variable')

        text = root.text.decode()

        for function_name, replacement in function_name_to_replacement.items():
            pattern = TextPreprocessor.RE_FUNCTION + re.escape(function_name) + TextPreprocessor.RE_WORD_BOUNDARY
            text = re.sub(pattern, r'\1\2' + replacement, text)

        for method_name, replacement in method_name_to_replacement.items():
            pattern = TextPreprocessor.RE_METHOD + re.escape(method_name) + TextPreprocessor.RE_WORD_BOUNDARY
            text = re.sub(pattern, r'\1\2' + replacement, text)

        for variable_name, replacement in variable_name_to_replacement.items():
            pattern = TextPreprocessor.RE_NEGATE_ALPHANUMERIC_AND_DOT + re.escape(variable_name) + TextPreprocessor.RE_WORD_BOUNDARY
            text = re.sub(pattern, replacement, text)

        return text
    
    @staticmethod
    def _replace_names_in_class(node: Node) -> str:
        class_names: list[str] = []
        property_names: list[str] = []

        root = node
        stack: list[Node] = [root]

        while len(stack) > 0:
            node = stack.pop()

            if TextPreprocessor._is_class_name(node):
                class_name = node.text.decode()
                class_names.append(class_name)

            if TextPreprocessor._is_property_name(node):
                property_name = node.text.decode()
                property_names.append(property_name)

            for child in reversed(node.children):
                stack.append(child)

        class_name_to_replacement = TextPreprocessor._map_name_to_replacement(class_names, 'class')
        property_name_to_replacement = TextPreprocessor._map_name_to_replacement_with_idx(property_names, 'property')

        text = root.text.decode()

        for class_name, replacement in class_name_to_replacement.items():
            pattern = TextPreprocessor.RE_CLASS + re.escape(class_name) + TextPreprocessor.RE_WORD_BOUNDARY
            text = re.sub(pattern, r'\1' + replacement, text)

        for property_name, replacement in property_name_to_replacement.items():
            pattern = TextPreprocessor.RE_WORD_BOUNDARY + re.escape(property_name) + TextPreprocessor.RE_WORD_BOUNDARY
            text = re.sub(pattern, replacement, text)

        return text

    @staticmethod
    def _is_variable_name(node: Node) -> bool | None:
        return node.type == 'identifier' \
            and node.parent.type in ['required_parameter', 'optional_parameter', 'variable_declarator', 'for_in_statement']

    @staticmethod
    def _is_property_name(node: Node) -> bool | None:
        return node.type == 'property_identifier' \
            and node.parent.type in ['public_field_definition']

    @staticmethod
    def _is_class_name(node: Node) -> bool | None:
        return node.type == 'type_identifier' \
            and node.parent.type in ['class_declaration']

    @staticmethod
    def _is_function_name(node: Node) -> bool | None:
        return node.type == 'identifier' \
            and node.parent.type in ['function_declaration', 'function_signature']

    @staticmethod
    def _is_method_name(node: Node) -> bool | None:
        return node.type == 'property_identifier' \
            and node.parent.type in ['method_definition', 'method_signature']
    
    @staticmethod
    def _map_name_to_replacement(names: list[str], tag: str) -> dict[str, str]:
        name_to_replacement: dict[str, str] = {}
        for name in names:
            name_to_replacement[name] = f'_{tag}_'

        return name_to_replacement
    
    @staticmethod
    def _map_name_to_replacement_with_idx(names: list[str], tag: str) -> dict[str, str]:
        name_to_replacement: dict[str, str] = {}
        for idx, name in enumerate(names):
            name_to_replacement[name] = f'_{tag}{idx}_'

        return name_to_replacement
    
    @staticmethod
    def _extract_code_tree(text: str) -> Tree:
        code_parser = Parser()
        code_parser.set_language(TextPreprocessor._load_typescript_language())
        tree = code_parser.parse(bytes(text, "utf8"))
        return tree

    @staticmethod
    def _load_typescript_language() -> Language:
        Language.build_library('build/my-languages.so', ['tree-sitter-typescript/typescript'])
        return Language('build/my-languages.so', 'typescript')