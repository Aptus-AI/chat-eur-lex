import glob
import json
import os.path

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import UnstructuredXMLLoader

import argparse
import yaml
from langchain_core.documents import Document
from lxml import etree

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

namespaces = {
    'akn': 'http://docs.oasis-open.org/legaldocml/ns/akn/3.0',
}

eurovoc_db = set()
for filename in glob.glob(config["akoma_ntoso_dataset_path"]):
    tree = etree.parse(filename)
    uri = tree.xpath('//akn:FRBRuri', namespaces=namespaces)[0].get('value')
    eurovoc = [element.get('showAs') for element in tree.xpath('//akn:keyword', namespaces=namespaces)]
    eurovoc_db.update(eurovoc)
    documents = []
    for element in tree.xpath(
        "//akn:article",
        namespaces=namespaces
    ):
        article_id = element.get('eId')
        content = "\n".join([
            ' '.join(paragraph.itertext()) for paragraph in element.xpath('.//akn:p', namespaces=namespaces)
        ])
        documents.append(Document(
            page_content=content,
            metadata={
                "article_id": article_id,
                "uri": uri,
                "eurovoc": eurovoc
            }
        ))
    db = vectoreDB_class.from_documents(documents, embedder, **config["vectorDB"]["kwargs"])

with open('eurovoc.json', 'w') as outfile:
    json.dump(list(eurovoc_db), outfile)

