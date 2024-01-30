from relativisticpy.parsers.types.position import Position

def display_error_with_code(code_str, error_line, error_char, error_message):

    # Split the code string into lines, preserving empty lines
    lines = code_str.splitlines(keepends=True)
    
    # Adjust error_line to 0-indexed for list access
    error_line_idx = error_line - 1

    # Ensure the error line number is within the range of the code lines
    if 0 <= error_line_idx < len(lines):
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


class Error:
    def __init__(self, pos_start: Position, pos_end: Position, error_name: str, details: str, source_code: str = None):
        self.pos_start = pos_start
        self.pos_end = pos_end
        self.error_name = error_name
        self.details = details
        self.source_code = source_code

    def as_string(self):
        result = f'{self.error_name}: {self.details}\n '
        result += f'Object Location: \n From Line {self.pos_start.line}, char {self.pos_start.character} \n to Line {self.pos_end.line}, char {self.pos_end.character}'
        if self.source_code:
            return display_error_with_code(self.source_code, self.pos_end.line, self.pos_end.character, result)
        else:
            return result
    
class IllegalCharacterError(Error):
    def __init__(self, pos_start, pos_end, details, source_code: str = None):
        super().__init__(pos_start, pos_end, 'Illegal Character', details, source_code)

class IllegalSyntaxError(Error):
    def __init__(self, pos_start, pos_end, details='', source_code: str = None):
        super().__init__(pos_start, pos_end, 'Invalid Syntax', details, source_code)

class IllegalAssignmentError(Error):
    def __init__(self, pos_start, pos_end, details='', source_code: str = None):
        super().__init__(pos_start, pos_end, 'Illegal Assignment or Definition', details, source_code)