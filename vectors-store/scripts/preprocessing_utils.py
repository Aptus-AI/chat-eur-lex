from bs4 import BeautifulSoup
import regex as re

def remove_html_lines(input_string):
    # Split the input string into lines
    lines = re.split(r'\n+', input_string)

    # Filter out lines that do not contain HTML tags
    filtered_lines = []
    for line in lines:
        soup = BeautifulSoup(line, 'html.parser')
        if not soup.find():
            filtered_lines.append(line)

    # Join the filtered lines back into a single string
    output_string = '\n'.join(filtered_lines)

    return output_string

