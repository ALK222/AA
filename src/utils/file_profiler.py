import ast
import sys


def get_function_lines(file_path):
    """
    Get the beginning and end line numbers of every function in a Python file.

    Args:
    - file_path (str): Path to the Python file.

    Returns:
    - dict: Dictionary where keys are function names and values are tuples of (begin_line, end_line).
    """
    function_lines = {}

    with open(file_path, 'r') as f:
        tree = ast.parse(f.read(), filename=file_path)

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            function_name = node.name
            begin_line = node.lineno
            end_line = node.end_lineno
            function_lines[function_name] = (begin_line, end_line)

    return function_lines


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    function_lines = get_function_lines(file_path)
    for function_name, (begin_line, end_line) in function_lines.items():
        print(
            f"Function '{function_name}' begins at line {begin_line} and ends at line {end_line}.")
