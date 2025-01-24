import nbformat
from black import format_str, FileMode
import sys
import os

def format_notebook(notebook_path, output_path=None, line_length=88):
    # Check if the file exists
    if not os.path.exists(notebook_path):
        print(f"Error: File '{notebook_path}' does not exist.")
        sys.exit(1)
    
    # Load the notebook file
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)

    # Format all code cells
    for cell in notebook.cells:
        if cell.cell_type == "code" and cell.source:  # Only format code cells
            try:
                cell.source = format_str(cell.source, mode=FileMode(line_length=line_length))
            except Exception as e:
                print(f"Skipping formatting for a cell due to an error: {e}")

    # Set default output path if not provided
    if not output_path:
        output_path = f"formatted_{os.path.basename(notebook_path)}"

    # Save the formatted notebook
    with open(output_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    print(f"Formatted notebook saved as '{output_path}'")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python format_notebook.py <notebook_path> [output_path] [line_length]")
        sys.exit(1)

    notebook_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    line_length = int(sys.argv[3]) if len(sys.argv) > 3 else 88

    format_notebook(notebook_path, output_path, line_length)