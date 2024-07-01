from langchain_community.vectorstores import Qdrant
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables.base import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.messages import AIMessage
from typing import List
from chat_utils import get_init_modules, SYSTEM_PROMPT, SYSTEM_PROMPT_LOOP, ContextInput, Answer, get_vectorDB_module
from langchain_core.documents.base import Document


class EurLexChat:
    def __init__(self, config: dict):
        self.config = config
        self.max_history_messages = self.config["max_history_messages"]
        self.use_functions = (
            'use_context_function' in config["llm"] and 
            config["llm"]["use_context_function"] and 
            config["llm"]["class"] == "ChatOpenAI")

        self.embedder, self.llm, self.chatDB_class, self.retriever = get_init_modules(config)
        self.max_context_size = config["llm"]["max_context_size"]

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])

        self.prompt_loop = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT_LOOP),
            ("human", "History: {history}. Message:"),
        ])

        self.chain_loop_answer = ( self.prompt_loop | self.llm )

        if self.use_functions: 

            GET_CONTEXT_TOOL = StructuredTool.from_function(
                func=self.get_context,
                name="get_context",
                description="To be used whenever the provided context is empty or the user changes the topic of the conversation and you need the context for the topic. " +
                "This function must be called only when is strictly necessary. " +
                "This function must not be called if you already have the information to answer the user. ",
                args_schema=ContextInput
            )

            # self.llm = self.llm.bind(tools=[convert_to_openai_tool(GET_CONTEXT_TOOL)])
            self.llm_with_functions = self.llm.bind(tools=[convert_to_openai_tool(GET_CONTEXT_TOOL)])
            
            chain = self.prompt | RunnableLambda(self._resize_history) | self.llm_with_functions
        else:
            chain = self.prompt | RunnableLambda(self._resize_history) | self.llm
        
        self.chain_with_history = RunnableWithMessageHistory(
            chain,
            self.get_chat_history,
            input_messages_key="question",
            history_messages_key="history",
        )

        self.relevant_documents_pipeline = ( self.retriever | self._parse_documents )


    def _resize_history(self, input_dict):
        """
        Resize the message history.

        Args:
            input_dict: The llm input containing the message history.

        Returns:
            dict: The resized version of the input_dict.
        """

        messages = input_dict.messages
        if (len(messages) - 2) > self.max_history_messages:
            messages = [messages[0]] + messages[-(self.max_history_messages +1):]
            input_dict.messages = messages
        return input_dict


    def get_chat_history(self, session_id: str):
        """
        Retrieve chat history instance for a specific session ID.

        Args:
            session_id (str): The unique identifier for the session.

        Returns:
            Chat history object: An instance of the appropriate chat history class.
        """

        kwargs = self.config["chatDB"]["kwargs"]
        if self.config["chatDB"]["class"] == 'FileChatMessageHistory':
            file_path = f"{kwargs['output_path']}/{session_id}.json"
            return self.chatDB_class(file_path=file_path)
        else:
            return self.chatDB_class(session_id=session_id, **kwargs)


    def _parse_documents(self, docs: List[Document]) -> List[dict]:
        """
        Parse a list of documents into a standardized format.

        Args:
            docs (List[Document]): A list of documents to parse.

        Returns:
            List[dict]: A list of dictionaries, each containing parsed information from the input documents.
        """

        parsed_documents = []

        for doc in docs:
            parsed_documents.append({
                'text': doc.page_content,
                'source': doc.metadata["source"],
                '_id': doc.metadata["_id"]
            })
        return parsed_documents


    def _format_context_docs(self, context_docs: List[dict]) -> str:
        """
        Format a list of documents into a single string.

        Args:
            context_docs (List[dict]): A list of dictionaries containing text from context documents.

        Returns:
            str: A string containing the concatenated text from all context documents.
        """

        context_str = ''
        for doc in context_docs:
            context_str += doc['text'] + "\n\n"
        return context_str


    def get_relevant_docs(self, question:str) -> List[dict]:
        """
        Retrieve relevant documents based on a given question.

        Args:
            question (str): The question for which relevant documents are retrieved.

        Returns:
            List[dict]: A list of relevant documents.
        """

        docs = self.relevant_documents_pipeline.invoke(question)
        return docs


    def get_context(self, text:str) -> str:
        """
        Retrieve context for a given text.

        Args:
            text (str): The text for which context is retrieved.

        Returns:
            str: A formatted string containing the relevant documents texts.
        """

        docs = self.get_relevant_docs(text)
        return self._format_context_docs(docs)


    def _remove_last_messages(self, session_id:str, n:int) -> None:
        """
        Remove last n messages from the chat history of a specific session.

        Args:
            session_id (str): The session ID for which messages are removed.
            n (int): The number of last messages to remove.
        """
        chat_history = self.get_chat_history(session_id=session_id)
        message_history = chat_history.messages
        chat_history.clear()
        message_history = message_history[:-n]
        for message in message_history:
            chat_history.add_message(message)


    def _format_history(self, session_id:str) -> str:
        """
        Format chat history for a specific session into a string.

        Args:
            session_id (str): The session ID for which the chat history is formatted.

        Returns:
            str: A formatted string containing the chat history for the specified session.
        """

        chat_history = self.get_chat_history(session_id).messages
        formatted_history = ""
        for message in chat_history:
            formatted_history += f"{message.type}: {message.content}\n\n"
        return formatted_history


    def _resize_context(self, context_docs:List[dict]) -> List[dict]:
        """
        Resize the dimension of the context in terms of number of tokens.
        If the concatenation of document text exceeds max_context_size,
        the document text is cut off to meet the limit.

        Args:
            context_docs (List[dict]): List of formatted documents.

        Returns:
            List[dict]: Returns the list of resized documents.
        """
        lengths = [self.llm.get_num_tokens(doc['text']) for doc in context_docs]
        resized_contexts = []
        total_len = 0
        for i, l in enumerate(lengths):
            if l + total_len <= self.max_context_size:
                resized_contexts.append(context_docs[i])
                total_len += l
        return resized_contexts
    
    def get_answer(self, session_id:str, question:str, context_docs:List[dict], from_tool:bool=False) -> Answer:
        """
        Get an answer to a question of a specific session, considering context documents and history messages.

        Args:
            session_id (str): The session ID for which the answer is retrieved.
            question (str): The new user message.
            context_docs (List[dict]): A list of documents used as context to answer the user message.
            from_tool (bool, optional): Whether the question originates from a tool. Defaults to False.

        Returns:
            Answer: An object containing the answer along with a new list of context documents 
                if those provided are insufficient to answer the question.

        """
        resized_docs = self._resize_context(context_docs)
        context = self._format_context_docs(resized_docs)

        result = self.chain_with_history.invoke(
            {"context": context, "question": question},
            config={"configurable": {"session_id": session_id}}
        )

        if self.use_functions and len(result.additional_kwargs) > 0:
            if from_tool:
                self._remove_last_messages(session_id=session_id, n=1)
                history = self._format_history(session_id)
                result = self.chain_loop_answer.invoke({'history': history})
                self.get_chat_history(session_id=session_id).add_message(AIMessage(result.content))
                return Answer(answer=result.content, status=-1)
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


class EurLexChatAkn(EurLexChat):
    def _parse_documents(self, docs: List[Document]) -> List[dict]:
        """
        Parse a list of documents into a standardized format.

        Args:
            docs (List[Document]): A list of documents to parse.

        Returns:
            List[dict]: A list of dictionaries, each containing parsed information from the input documents.
        """

        parsed_documents = []

        for doc in docs:
            parsed_documents.append({
                'text': doc.page_content,
                'source': doc.metadata["uri"],
                '_id': doc.metadata["uri"] + doc.metadata["article_id"]
            })
        return parsed_documents

    def get_relevant_docs(self, question: str, eurovoc: str = None) -> List[dict]:
        """
        Retrieve relevant documents based on a given question.

        Args:
            question (str): The question for which relevant documents are retrieved.
            eurovoc (str): The Eurovoc to be used as filter

        Returns:
            List[dict]: A list of relevant documents.
        """
        if eurovoc:
            retriever = get_vectorDB_module(
                self.config['vectorDB'], self.embedder, metadata={'filter': {'eurovoc': ''}}
            )
            relevant_documents_pipeline_with_filter = (retriever | self._parse_documents)
            docs = relevant_documents_pipeline_with_filter.invoke(
                question
            )
        else:
            docs = self.relevant_documents_pipeline.invoke(question)
        return docs