from langchain_community.vectorstores import Qdrant
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables.base import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.messages import AIMessage
from dataclasses import dataclass
from typing import List, Optional

SYSTEM_PROMPT = '''Sei un assistente che deve rispondere e conversare con l'utente utilizzando il contesto fornito di seguito.
            Se il contesto non è sufficiente per rispondere non inventare una risposta. Rispondi nella stessa lingua con cui parla l'utente.\n\n ### Contesto:\n {context}'''

ANSWER_CONTEXT_LOOP = "Scusa ma utilizzando il contesto fornito non ti riesco a rispondere"
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

class EurLexChat:
    def __init__(self, config: dict):
        self.config = config
        self.max_history_messages = self.config["max_history_messages"]
        db_config = self.config['vectorDB']
        self.use_functions = 'use_context_function' in config["llm"] and config["llm"]["use_context_function"]
        self.embedder = self._get_instance_dynamic_class(
            lib_path='langchain_community.embeddings',
            class_name=config["embeddings"]["class"],
            **config["embeddings"]["kwargs"]
        )

        self.llm = self._get_instance_dynamic_class(
            lib_path='langchain_community.chat_models',
            class_name=config["llm"]["class"],
            **config["llm"]["kwargs"]
        )

        mod_chat = __import__("langchain_community.chat_message_histories", fromlist=[config["chatDB"]["class"]])
        self.chatDB_class = getattr(mod_chat, config["chatDB"]["class"])

        if db_config['class'] == 'Qdrant':
            from qdrant_client import QdrantClient
            from langchain_community.vectorstores import Qdrant

            client_args = ["url", "port"]
            client_kwargs = {k:v for k,v in db_config['kwargs'].items() if k in client_args }
            db_kwargs = {k:v for k,v in db_config['kwargs'].items() if k not in client_kwargs }

            client = QdrantClient(**client_kwargs)

            retriever = Qdrant(
                client, embeddings=self.embedder, **db_kwargs).as_retriever(
                    search_type=db_config['retriever_args']["search_type"],
                    search_kwargs=db_config['retriever_args']["search_kwargs"]
            )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])
        if self.use_functions: 

            GET_CONTEXT_TOOL = StructuredTool.from_function(
                func=self.get_context,
                name="get_context",
                description='To be used whenever the user change the subject of the conversation and you need the context for the new subject. ' +
                'This function must not be called right after the first message of the user or if you already have the information to answer. ' + 
                'Use this function only when is strictly necessary',
                args_schema=ContextInput
            )
            #TODO: let the use of the functions applicable to all llms and not only to openai
            self.llm = self.llm.bind(tools=[convert_to_openai_tool(GET_CONTEXT_TOOL)])
            
            chain = self.prompt | RunnableLambda(self._resize_history) | self.llm
        else:
            chain = self.prompt | RunnableLambda(self._resize_history) | self.llm
        
        self.chain_with_history = RunnableWithMessageHistory(
            chain,
            self.get_chat_history,
            input_messages_key="question",
            history_messages_key="history",
        )
        
        self.relevant_documents_pipeline = ( retriever | self._parse_documents )


    def _get_instance_dynamic_class(self, lib_path, class_name, **kwargs):
        mod = __import__(lib_path, fromlist=[class_name])
        dynamic_class = getattr(mod, class_name)
        return dynamic_class(**kwargs)


    def _resize_history(self, input_dict):
        messages = input_dict.messages

        if (len(messages) - 2) > self.max_history_messages:
            messages = [messages[0]] + messages[-(self.max_history_messages +1):]
            input_dict.messages = messages
        return input_dict


    def get_chat_history(self, session_id: str):
        kwargs = self.config["chatDB"]["kwargs"]
        if self.config["chatDB"]["class"] == 'FileChatMessageHistory':
            file_path = f"{kwargs['output_path']}/{session_id}.json"
            return self.chatDB_class(file_path=file_path)
        else:
            return self.chatDB_class(session_id=session_id, **kwargs)


    def _parse_documents(self, docs):
        parsed_documents = []

        for doc in docs:
            parsed_documents.append({
                'text': doc.page_content,
                'source': doc.metadata["source"]
            })
        return parsed_documents


    def _format_context_docs(self, context_docs):
        context_str = ''
        for doc in context_docs:
            context_str += doc['text'] + "\n\n"
        return context_str


    def get_relevant_docs(self, question):
        docs = self.relevant_documents_pipeline.invoke(question)
        return docs

    def get_context(self, text):
        docs = self.get_relevant_docs(text)
        return self._format_context_docs(docs)

    def _remove_last_messages(self, session_id, n):
        chat_history = self.get_chat_history(session_id=session_id)
        message_history = chat_history.messages
        chat_history.clear()
        message_history = message_history[:-n]
        for message in message_history:
            chat_history.add_message(message)


    def get_answer(self, session_id, question, context_docs, from_tool=False):
         
        context = self._format_context_docs(context_docs)

        result = self.chain_with_history.invoke(
            {"context": context, "question": question},
            config={"configurable": {"session_id": session_id}}
        )

        if len(result.additional_kwargs) > 0:
            # TODO: se per due volte di suguito viene triggherata la regola, 
            # oppure dire al modello di generare una risposta per in cui deve dire di 
            # riformulare la domanda perchè non trova informazioni a riguardo
            if from_tool:
                self._remove_last_messages(session_id=session_id, n=1)
                self.get_chat_history(session_id=session_id).add_message(AIMessage(ANSWER_CONTEXT_LOOP))
                return Answer(answer=ANSWER_CONTEXT_LOOP, status=-1)
            text = eval(result.additional_kwargs['tool_calls'][0]['function']['arguments'])['text']
            new_docs = self.get_relevant_docs(text)
            self._remove_last_messages(session_id=session_id, n=2)

            result = self.get_answer(
                session_id=session_id,
                question=question,
                context_docs=new_docs,
                from_tool=True
            )
            if result.status == 1:
                return Answer(answer=result.answer, new_documents=new_docs)
            else:
                return Answer(answer=result.answer)        
        return Answer(answer=result.content)

# TODO: mettere prompt in inglese
# TODO: Aggiungere le referenze alla risposta?

