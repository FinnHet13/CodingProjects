import os
import re
import nbformat
from collections import defaultdict
import subprocess

def get_installed_version(package):
    """
    Get the installed version of a package using pip.
    """
    try:
        result = subprocess.run(["pip", "show", package], capture_output=True, text=True, check=True)
        for line in result.stdout.splitlines():
            if line.startswith("Version:"):
                return line.split()[1]
    except subprocess.CalledProcessError:
        return None

def extract_dependencies_from_notebook(notebook_path):
    """
    Extract dependencies and their versions from a Jupyter notebook.
    """
    dependencies = defaultdict(lambda: None)

    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = nbformat.read(f, as_version=4)

    for cell in notebook.get('cells', []):
        if cell['cell_type'] == 'code':
            # Search for pip install commands
            pip_matches = re.findall(r'!pip install ([^\s]+)(==[^\s]+)?', cell['source'])
            for match in pip_matches:
                package = match[0]
                version = match[1].lstrip('==') if match[1] else None
                dependencies[package] = version

            # Search for import statements
            import_matches = re.findall(r'^import ([^\s]+)|^from ([^\s]+)', cell['source'], re.MULTILINE)
            for match in import_matches:
                package = match[0] or match[1]
                if package not in dependencies:
                    dependencies[package] = None

    return dependencies

def extract_dependencies_from_folder(folder_path):
    """
    Extract dependencies and their versions from all Jupyter notebooks in a folder.
    """
    all_dependencies = defaultdict(lambda: None)

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.ipynb'):
                notebook_path = os.path.join(root, file)
                notebook_dependencies = extract_dependencies_from_notebook(notebook_path)
                for package, version in notebook_dependencies.items():
                    if package not in all_dependencies or not all_dependencies[package]:
                        all_dependencies[package] = version

    # Retrieve installed versions for packages without specified versions
    for package in all_dependencies:
        if not all_dependencies[package]:
            all_dependencies[package] = get_installed_version(package)

    return all_dependencies

def generate_conda_yaml(dependencies, output_file):
    """
    Generate a conda environment YAML file from the extracted dependencies.
    """
    yaml_content = ["name: extracted_environment", "channels:", "  - defaults", "dependencies:"]

    for package, version in sorted(dependencies.items()):
        if version:
            yaml_content.append(f"  - {package}={version}")
        else:
            yaml_content.append(f"  - {package}")

    # Ensure pip is included
    yaml_content.append("  - pip")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(yaml_content))

    print(f"YAML file generated at: {output_file}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Extract dependencies from Jupyter notebooks and create a conda environment YAML file.")
    parser.add_argument("folder_path", help="Path to the folder containing Jupyter notebooks")
    parser.add_argument("output_file", help="Path for the output .yaml file (e.g., environment.yaml)")

    args = parser.parse_args()

    print("Extracting dependencies from notebooks...")
    dependencies = extract_dependencies_from_folder(args.folder_path)

    print("Generating conda environment YAML file...")
    generate_conda_yaml(dependencies, args.output_file)

if __name__ == "__main__":
    main()
