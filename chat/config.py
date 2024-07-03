import os
import yaml


# Read config file
if os.path.exists('config.yaml'):
    with open('config.yaml', 'r') as file:
        CONFIG = yaml.safe_load(file)
else:
    raise FileNotFoundError('config.yml not found Aborting!')

OPENAI_ORG_KEY = os.getenv("OPENAI_ORG_KEY", "")
OPENAI_KEY = os.getenv("OPENAI_KEY", "")
QDRANT_URL = os.getenv("url", CONFIG["vectorDB"]["kwargs"].get("url", ""))
QDRANT_KEY = os.getenv("qdrant_key", CONFIG["vectorDB"]["kwargs"].get("api_key", ""))

UI_USER = os.getenv("user", "admin")
UI_PWD = os.getenv("pwd", "admin")

CONFIG["embeddings"]["kwargs"]["openai_api_key"] = OPENAI_KEY
CONFIG["embeddings"]["kwargs"]["openai_organization"] = OPENAI_ORG_KEY
CONFIG["llm"]["kwargs"]["openai_api_key"] = OPENAI_KEY
CONFIG["llm"]["kwargs"]["openai_organization"] = OPENAI_ORG_KEY
CONFIG["vectorDB"]["kwargs"]["url"] = QDRANT_URL
CONFIG["vectorDB"]["kwargs"]["api_key"] = QDRANT_KEY

# if the history should be stored on AWS DynamoDB
# otherwise it will be stored on local FS to the output_path defined in the config.yaml file
if CONFIG['chatDB']['class'] == 'DynamoDBChatMessageHistory':
    CHATDB_TABLE_NAME = os.getenv("CHATDB_TABLE_NAME", CONFIG["chatDB"]["kwargs"].get("table_name", "ChatEurlexHistory"))
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", CONFIG["chatDB"]["kwargs"].get("aws_access_key_id", ""))
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", CONFIG["chatDB"]["kwargs"].get("aws_secret_access_key", ""))
    CONFIG["chatDB"]["kwargs"]["table_name"] = CHATDB_TABLE_NAME
    CONFIG["chatDB"]["kwargs"]["aws_access_key_id"] = AWS_ACCESS_KEY_ID
    CONFIG["chatDB"]["kwargs"]["aws_secret_access_key"] = AWS_SECRET_ACCESS_KEY
