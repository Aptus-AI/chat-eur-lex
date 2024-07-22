from dataclasses import dataclass
from typing import Optional, List
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import ConfigurableField
from langchain_core.runnables.base import RunnableLambda
from operator import itemgetter

SYSTEM_PROMPT = (
    "You are an assistant  specialized in the legal and compliance field who must answer and converse with the user using the context provided. " +
    "When you answer the user, if it is relevant, cite the laws and articles you are referring to. NEVER mention the use of context in your answers. " +
    "If the user asks for a definition, report exactly the content of the context, do not paraphrase the text." +
    "If you believe the question cannot be answered from the given context, do not make up an answer. Answer in the same language the user is speaking.\n\n ### Context:\n {context}"
)

SYSTEM_PROMPT_LOOP = (
    "You are an assistant who must inform the user that you do not have enough information to answer and ask if the user can provide you with additional information. " +
    "This answer, must be adapted to the conversation that occurred with the user that is provided to you. Just write down the answer "
)

@dataclass
class Answer():
    answer: str
    new_documents: Optional[List] = None
    status: Optional[int] = 1

class ContextInput(BaseModel):
    text: str = Field(
        title="Text",
        description="Self-explanatory summary describing what the user is asking for"
        )

def get_instance_dynamic_class(lib_path:str, class_name:str, **kwargs):
    """
    Instantiate a dynamically imported class from a given library path and class name.

    Args:
        lib_path (str): The path to the library/module containing the class.
        class_name (str): The name of the class to instantiate.
        **kwargs: Additional keyword arguments to pass to the class constructor.

    Returns:
        An instance of the dynamically imported class initialized with the provided arguments.
    """

    mod = __import__(lib_path, fromlist=[class_name])
    dynamic_class = getattr(mod, class_name)
    return dynamic_class(**kwargs)


def get_init_modules(config):
    embedder = get_instance_dynamic_class(
        lib_path='langchain_community.embeddings',
        class_name=config["embeddings"]["class"],
        **config["embeddings"]["kwargs"]
    )

    llm = get_instance_dynamic_class(
        lib_path='langchain_community.chat_models',
        class_name=config["llm"]["class"],
        **config["llm"]["kwargs"]
    )

    mod_chat = __import__("langchain_community.chat_message_histories",
                          fromlist=[config["chatDB"]["class"]])
    chatDB_class = getattr(mod_chat, config["chatDB"]["class"])
    retriever, retriever_chain = get_vectorDB_module(config['vectorDB'], embedder)

    return embedder, llm, chatDB_class, retriever, retriever_chain

def get_vectorDB_module(db_config, embedder):
    mod_chat = __import__("langchain_community.vectorstores",
                          fromlist=[db_config["class"]])
    vectorDB_class = getattr(mod_chat, db_config["class"])

    if db_config["class"] == 'Qdrant':
        from qdrant_client import QdrantClient
        import inspect

        # Get QdrantClient init parameters name from signature
        signature_params = inspect.signature(QdrantClient.__init__).parameters.values()
        params_to_exclude = ['self', 'kwargs']
        client_args = [el.name for el in list(signature_params) if el.name not in params_to_exclude]

        client_kwargs = {k: v for k,
                         v in db_config['kwargs'].items() if k in client_args}
        db_kwargs = {
            k: v for k, v in db_config['kwargs'].items() if k not in client_kwargs}

        client = QdrantClient(**client_kwargs)

        retriever = vectorDB_class(
            client, embeddings=embedder, **db_kwargs).as_retriever(
                search_type=db_config["retriever_args"]["search_type"],
                search_kwargs={**db_config["retriever_args"]["search_kwargs"]}
        )

    else:
        retriever = vectorDB_class(embeddings=embedder, **db_config["kwargs"]).as_retriever(
            search_type=db_config["retriever_args"]["search_type"],
            search_kwargs=db_config["retriever_args"]["search_kwargs"]
        )

    retriever = retriever.configurable_fields(
        search_kwargs=ConfigurableField(
            id="search_kwargs",
            name="Search Kwargs",
            description="The search kwargs to use. Includes dynamic category adjustment.",
        )
    )

    chain = ( RunnableLambda(lambda x: x['question']) | retriever)

    if db_config.get("rerank"):
        if db_config["rerank"]["class"] == "CohereRerank":
            module_compressors = __import__("langchain.retrievers.document_compressors",
                                        fromlist=[db_config["rerank"]["class"]])
            rerank_class = getattr(module_compressors, db_config["rerank"]["class"])
            rerank = rerank_class(**db_config["rerank"]["kwargs"])

            chain = ({
                "docs": chain,
                "query": itemgetter("question"),
                } | (RunnableLambda(lambda x: rerank.compress_documents(x['docs'], x['query'])))
                    )
        else:
            raise NotImplementedError(db_config["rerank"]["class"])
    return retriever, chain

