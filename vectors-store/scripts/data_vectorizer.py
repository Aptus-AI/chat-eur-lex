from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from preprocessing_utils import remove_html_lines
import argparse
import yaml
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

# Load and preprocess documents
loader = DirectoryLoader(
    config['data_path'],
    glob="**/*.txt",
    show_progress=True,
    use_multithreading=True,
    max_concurrency=8
    )

docs = loader.load()
for i in range(len(docs)):
    docs[i].page_content = remove_html_lines(docs[i].page_content)

# Create TextSplitter according to configs and chunk documents
if splitter_class_name == 'LineSplitter':
    # Text splitter chunking by number of rows (defined as newlines)

    kwargs = config["splitter"]["kwargs"]
    separator = kwargs["separator"] if "separator" in kwargs else "\n"
    text_splitter = CharacterTextSplitter(
        length_function=lambda x: 0 if x == separator else 1,
        **config["splitter"]["kwargs"]
    )
elif splitter_class_name == 'OpenAICharacterSplitter':
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        **config["splitter"]["kwargs"]
    )
elif splitter_class_name == 'HuggingfaceCharacterSplitter':
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["splitter"]['tokenizer_name'])
    text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer = tokenizer,
        **config["splitter"]["kwargs"]
    )
else:
    mod_splitter = __import__('langchain.text_splitter', fromlist=[config["splitter"]["class"]])
    splitter_class = getattr(mod_splitter, config["splitter"]["class"])
    text_splitter = splitter_class(**config["splitter"]["kwargs"])

chunked_documents = text_splitter.split_documents(docs)

# Embed chunk and store in the vectorDB
db = vectoreDB_class.from_documents(chunked_documents, embedder, **config["vectorDB"]["kwargs"])
