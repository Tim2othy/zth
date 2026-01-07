"""Script to clean text files by removing unwanted characters.
Saves cleaned files to the 'clean' directory. Which is gitignored. so the data isn't saved twice in the repo.
"""

file1 = "unsong.txt"
file2 = "moby-dick.txt"
file3 = "shakespeare.txt"
files = [file1, file2, file3]


# Function to filter text, keeping only allowed characters
def filter_text(text):
    # Keep alphanumeric, whitespace, and basic punctuation
    allowed_chars = set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789  !$&-?.,;:'\n\t"
    )
    return "".join(char for char in text if char in allowed_chars)


# Save clean versions
for file in files:
    with open("clean/clean_" + file, "w", encoding="utf-8") as outfile:
        with open(file, "r", encoding="utf-8") as f1:
            filtered_content = filter_text(f1.read())
            filtered_content = filtered_content.lower()
            outfile.write(filtered_content)


# Concatinate the three clean files
with open("clean/books_cat.txt", "w", encoding="utf-8") as outfile:
    for file in files:
        with open("clean/clean_" + file, "r", encoding="utf-8") as f:
            content = f.read()
            outfile.write(content)
            outfile.write("\n")
