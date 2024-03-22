from relativisticpy.interpreter.nodes.position import TokenPosition

def display_error_with_code(code_str, error_line, error_char, error_message):

    # Split the code string into lines, preserving empty lines
    lines = code_str.splitlines(keepends=True)
    
    # Adjust error_line to 0-indexed for list access
    error_line_idx = error_line - 1

    # Ensure the error line number is within the range of the code lines
    if 0 <= error_line_idx and error_line_idx < len(lines):
        error_line_content = lines[error_line_idx].rstrip('\n')
        
        # Display the error message
        message = f"{error_message}\n"

        # Display the line number and the line content
        message += f"Line {error_line}: {error_line_content}\n"

        # Indicate the position of the error with a caret
        message += (' ' * (len(f"Line {error_line}: ") + error_char - 1) + '^')
    else:
        message = "Error location is out of the code range."

    return message

class RelPyError(Exception):
    def __init__(self, pos: TokenPosition, error_name: str, details: str, source_code: str = None):
        self.pos_start = pos.start_pos
        self.pos_end = pos.end_pos
        self.error_name = error_name
        self.details = details
        self.source_code = source_code
        super().__init__(self.as_string())

    def as_string(self):
        result = f'{self.error_name}: {self.details}\n '
        result += f'Object Location: \n From Line {self.pos_start.line}, char {self.pos_start.character} \n to Line {self.pos_end.line}, char {self.pos_end.character}'
        if self.source_code:
            return display_error_with_code(self.source_code, self.pos_end.line, self.pos_end.character, result)
        else:
            return result
    

# General Language Errors

class IllegalCharacterError(RelPyError):
    def __init__(self, pos: TokenPosition, details, source_code: str = None):
        super().__init__(pos, 'Illegal Character', details, source_code)

class IllegalSyntaxError(RelPyError):
    def __init__(self, pos: TokenPosition, details='', source_code: str = None):
        super().__init__(pos, 'Invalid Syntax', details, source_code)

class IllegalAssignmentError(RelPyError):
    def __init__(self, pos: TokenPosition, details='', source_code: str = None):
        super().__init__(pos, 'Illegal Assignment or Definition', details, source_code)

# Interpreter Errors

# ...

# Tensor Errors

class IndicesError(RelPyError):
    def __init__(self, pos: TokenPosition, details, source_code: str = None):
        super().__init__(pos, 'Indices Error', details, source_code)

class TensorError(RelPyError):
    def __init__(self, pos: TokenPosition, details, source_code: str = None):
        super().__init__(pos, 'Tensor Error', details, source_code)