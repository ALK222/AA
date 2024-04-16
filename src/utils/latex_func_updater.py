import ast
import re
import sys
import json


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


def update_latex_file(latex_content, function_lines, python_file):
    """
    Update a LaTeX content with new line numbers for code references.

    Args:
    - latex_content (str): Content of the LaTeX file.
    - function_lines (dict): Dictionary containing function names and their line numbers.
    - python_file (str): Name of the Python file.

    Returns:
    - str: Updated LaTeX content.
    """
    updated_latex_content = latex_content

    for function_name, (begin_line, end_line) in function_lines.items():
        function_name_r = function_name.replace('_', '\\_')
        pattern = rf'\\lstinputlisting\[firstline=(\d+),lastline=(\d+), style=custompython\]{{[^}}]+}}\n\s*\\caption{{.*?}}\n\s*\\label{{fig:{function_name}}}'
        new_pattern = rf'\\lstinputlisting[firstline={begin_line},lastline={end_line}, style=custompython]{{../{python_file.split("/")[-1]}}}\n\\caption{{Función \\textit{{{function_name_r}}}}}\n\\label{{fig:{function_name}}}'

        match = re.search(pattern, latex_content)
        if match:
            print(match.end())
            print(f"Updating {function_name} in LaTeX file")
        else:
            print(f'match not found for {function_name}', pattern)
        updated_latex_content = re.sub(
            pattern, new_pattern, updated_latex_content)

    return updated_latex_content


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python combined_script.py <latex_file_path> <python_file>")
        sys.exit(1)

    latex_file_path = sys.argv[1]
    python_file = sys.argv[2]

    # Read LaTeX content from the file
    with open(latex_file_path, 'r') as f:
        latex_content = f.read()

    # Generate function lines from the Python file
    function_lines = get_function_lines(python_file)

    # Update LaTeX content with new function line numbers
    updated_latex_content = update_latex_file(
        latex_content, function_lines, python_file)

    # Write updated LaTeX content to the file
    with open(latex_file_path + ".new", 'w') as f:
        f.write(updated_latex_content)
