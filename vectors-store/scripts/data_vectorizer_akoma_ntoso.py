import glob
import argparse
import yaml
from langchain_core.documents import Document
from lxml import etree
import re
from tqdm import tqdm
import numpy as np
import tiktoken
from functools import reduce
import spacy
from typing import List, Union, Tuple

class SpacyTokenizer:
    def __init__(self, nlp) -> "SpacyTokenizer":
        self.nlp = nlp
    def encode(self, text):
        doc = nlp(text)
        return [token.text for token in doc]
    
def chunk_long_paragraphs(paragraph_content:List[str], 
                          paragraph_tokens_size:List[int], 
                          threshold:int, 
                          tokenizer:Union[tiktoken.core.Encoding,SpacyTokenizer], 
                          nlp:spacy.Language, 
                          recursion_flag:bool=False) -> Tuple[List[str], List[List[Union[int, str]]]]:
    """
    Splits long paragraphs into smaller chunks based on a token size threshold.

    Args:
        paragraph_content (List[str]): List of paragraphs to be chunked.
        paragraph_tokens_size (List[int]): List of token counts for each paragraph.
        threshold (int): Maximum allowed token size for each chunk.
        tokenizer (Union[tiktoken.core.Encoding,SpacyTokenizer]): Tokenizer instance used to encode the text.
        nlp (spacy.Language): spaCy language model instance for sentence segmentation.
        recursion_flag (bool): Flag indicating whether the function is being called recursively.

    Returns:
        tuple:
            - List of str: List of chunked paragraph contents.
            - List of list of int/str: List of lists indicating the original paragraph indices 
              or sentence indices for each chunk.

    Example:
        >>> paragraphs = ["This is a very long paragraph that needs to be split.", "Short paragraph."]
        >>> token_counts = [50, 5]
        >>> threshold = 40
        >>> from transformers import AutoTokenizer
        >>> import spacy
        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        >>> nlp = spacy.load("en_core_web_sm")
        >>> chunked_paragraphs, chunked_indices = chunk_long_paragraphs(paragraphs, token_counts, threshold, tokenizer, nlp)
        >>> print(chunked_paragraphs)
        >>> print(chunked_indices)
    """
    # List to hold the concatenated paragraphs and their indices
    paragraph_contents = []
    paragraph_numbers = []
    current_chunk = []
    current_length = 0
    last_idx = 0

    for i, (text, length) in enumerate(zip(paragraph_content, paragraph_tokens_size)):
        if length > threshold and not recursion_flag:
            # If a paragraph is too long, break it down into sentences
            paragraph_sentences = [sent.text for sent in nlp(text).sents]
            sentences_lengths = [len(tokenizer.encode(sent)) for sent in paragraph_sentences]
            # Recursively chunk long sentences
            sentence_chunks, sentence_numbers = chunk_long_paragraphs(paragraph_sentences, sentences_lengths, threshold, tokenizer, nlp, recursion_flag=True)
            sentence_numbers = [[f"{i}.{sent_num}" for sent_num in chunk] for chunk in sentence_numbers]

            if current_chunk:
                # Add current chunk to the results before processing long paragraph
                paragraph_contents.append('\n'.join(current_chunk))
                paragraph_numbers.append(list(range(last_idx, i)))
                current_chunk = []
                current_length = 0

            paragraph_contents.extend(sentence_chunks)
            paragraph_numbers.extend(sentence_numbers)
            last_idx = i + 1

        else:
            # Add the paragraph to the current chunk if it fits
            if current_length + length + len(current_chunk) <= threshold:
                current_chunk.append(text)
                current_length += length
            else:
                # Save the current chunk if adding the next paragraph would exceed the threshold
                if current_chunk:
                    paragraph_contents.append('\n'.join(current_chunk))
                    paragraph_numbers.append(list(range(last_idx, i)))
                current_chunk = [text]
                current_length = length
                last_idx = i

    # Add any remaining chunk
    if current_chunk:
        paragraph_contents.append('\n'.join(current_chunk))
        paragraph_numbers.append(list(range(last_idx, len(paragraph_content))))

    return paragraph_contents, paragraph_numbers
        
parser = argparse.ArgumentParser(description="Preprocessing Chat-eur-lex service")
parser.add_argument('--config_path',
                    dest='config_path',
                    metavar='config_path',
                    type=str,
                    help='The path to the config file that contains all the settings for the preprocessing' ,
                    default='config.yaml')
args = parser.parse_args()

# Read config file
with open(args.config_path, 'r') as file:
    config = yaml.safe_load(file)

# Dynamic import of embeddings and vectorDB modules
mod_emb = __import__('langchain_community.embeddings', fromlist=[config["embeddings"]["class"]])
embeddings_class = getattr(mod_emb, config["embeddings"]["class"])

mod_db = __import__('langchain_community.vectorstores', fromlist=[config["vectorDB"]["class"]])
vectoreDB_class = getattr(mod_db, config["vectorDB"]["class"])
splitter_class_name = config["splitter"]["class"]

mod = __import__('langchain_community.embeddings', fromlist=[config["embeddings"]["class"]])
embeddings_class = getattr(mod, config["embeddings"]["class"])
embedder = embeddings_class(**config["embeddings"]["kwargs"])

# Stats
article_length_words = []
article_length_tokens = []

# Load spacy
try:
    nlp = spacy.load("en_core_web_sm")
except:
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Setup tokenizer
if config["embeddings"]["class"] == "OpenAIEmbeddings":
    tokenizer = tiktoken.encoding_for_model(config["embeddings"]["kwargs"]["model"])
else:
    tokenizer = SpacyTokenizer(nlp)

# Load document text size
threshold = config["vectorDB"]["akn_document_token_size"]
# Setup rules to clean texts
substitutions = [("\\t+", "\\t"), (r"\s+", " "), ("\\n+", "\\n")]

# VectorDB documents
documents = []

# Setup AKN namespace
namespaces = {
    'akn': 'http://docs.oasis-open.org/legaldocml/ns/akn/3.0',
}
# Loading documents - use glob to get all files in the directory
all_files = glob.glob(config["akoma_ntoso_dataset_path"])
# Loading documents - define a regular expression pattern to exclude corrigendum, ie. files ending with R(\d{1,})
filtered_files = [file for file in all_files if not re.compile(r'R\(\d{1,}\).xml').search(file)]

# Loop through files
for filename in tqdm(filtered_files, desc="Chunking"):
    tree = etree.parse(filename)
    uri = tree.xpath('//akn:FRBRuri', namespaces=namespaces)[0].get('value')
    language = tree.xpath('//akn:FRBRlanguage', namespaces=namespaces)[0].get('language')
    celex = [x.get('value') for x in tree.xpath('//akn:FRBRalias', namespaces=namespaces) if x.get('name').lower() == "celex"][0]
    eurovoc = [element.get('showAs') for element in tree.xpath('//akn:keyword', namespaces=namespaces)]
    pub_date = tree.xpath('//akn:publication', namespaces=namespaces)[0].get('date')
    year, month, day = pub_date.split("-")
    doc_number = tree.xpath('//akn:FRBRnumber', namespaces=namespaces)[0].get('value')

    # Loop over articles of the document
    for element in tree.xpath("//akn:article", namespaces=namespaces):
        article_id = element.get('eId')
        # Retrieve each paragraph text
        paragraph_content = [' '.join(paragraph.itertext()) for paragraph in element.xpath('.//akn:p', namespaces=namespaces)]
        # Apply all substitutions to each paragraph
        paragraph_content = [reduce(lambda t, s: re.sub(s[0], s[1], t), substitutions, p) for p in paragraph_content]
        paragrah_tokens_size = [len(tokenizer.encode(p)) for p in paragraph_content]
       
        # ---   
        # Concatenate the content of the paragraph included in the article to compute figures
        content = "\n".join([
            ' '.join(paragraph.itertext()) for paragraph in element.xpath('.//akn:p', namespaces=namespaces)
        ])
        # Update statistics
        article_length_words.append(len(content.split()))
        article_tokens = tokenizer.encode(content)
        article_length_tokens.append(len(article_tokens))
        # ---   

        # If the number of tokens of the current article is over the threshold
        #  then split the article into chunk of paragraphs.
        if np.sum(paragrah_tokens_size) > threshold:
            paragraph_contents, paragraph_numbers = chunk_long_paragraphs(paragraph_content, paragrah_tokens_size, threshold, tokenizer, nlp)
        else:
            paragraph_contents = ["\n".join(paragraph_content)]
            paragraph_numbers = [[n for n in range(len(paragraph_content))]]
            

        for i, text in enumerate(paragraph_contents):
            documents.append(Document(
                page_content=text,
                metadata={
                    "article_id": article_id,
                    "paragraphs": paragraph_numbers[i],
                    "source": uri,
                    "celex": celex,
                    "eurovoc": eurovoc,
                    "language": language,
                    "doc_number": doc_number,
                    "year": year,
                    "month": month,
                    "day": day,
                }
            ))

# Print statistics
print("Number of documents", len(filtered_files))
print("Number of chunks", len(documents))
print("Token based statistics")
print("Total tokens", np.sum(article_length_tokens))
print("AVG chunk length (tokens)", np.mean(article_length_tokens))
print("ST.d chunk length (tokens)", np.std(article_length_tokens))
print("Median chunk length (tokens)", np.median(article_length_tokens))
print(f"# texts over {threshold} tokens", np.sum(np.array(article_length_tokens) > threshold))

# Create the VectorDB collection
print('Embed and index...')
db = vectoreDB_class.from_documents(documents, embedder, **config["vectorDB"]["kwargs"])
print('Completed')