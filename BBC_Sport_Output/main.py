import re

def count_tokens(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            tokens = re.findall(r'\b\w+\b', text)  # Using regex to find words
            return len(tokens)
    except FileNotFoundError:
        print("File not found. Please provide a valid file path.")
        return 0

# Example usage
file_path = "./data.txt"  # Replace with your file path
num_tokens = count_tokens(file_path)
print(f"Number of tokens in the file: {num_tokens}")