import nbformat
import textwrap
import re

def advanced_wrap_notebook(notebook_path, width=88):
    nb = nbformat.read(notebook_path, as_version=4)
    for cell in nb.cells:
        if cell['cell_type'] == 'code':
            lines = cell['source'].split('\n')
            wrapped_lines = []
            
            for line in lines:
                # Preserve full-line comments and docstrings
                if line.strip().startswith(('#', '"""', "'''")):
                    wrapped_lines.append(line)
                    continue
                
                # Handle lines with inline comments
                comment_match = re.match(r'^(.*?)(\s*#.*)$', line)
                if comment_match:
                    code_part, comment_part = comment_match.groups()
                    
                    # Special handling for complex lines like DataFrame column definitions
                    if '==' in code_part or '=' in code_part:
                        wrapped_lines.append(line)
                    else:
                        # Wrap code part while preserving indentation
                        indent = len(line) - len(line.lstrip())
                        wrapped_code = textwrap.fill(
                            code_part.rstrip(), 
                            width=width-indent, 
                            break_long_words=False, 
                            break_on_hyphens=False
                        )
                        wrapped_line = line[:indent] + wrapped_code + comment_part
                        wrapped_lines.append(wrapped_line)
                else:
                    # Standard line wrapping
                    wrapped_lines.append(line)
            
            cell['source'] = '\n'.join(wrapped_lines)
    
    nbformat.write(nb, notebook_path)

if __name__ == '__main__':
    import sys
    advanced_wrap_notebook(sys.argv[1])