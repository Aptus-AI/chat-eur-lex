import boto3
from langchain_community.vectorstores import Qdrant
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables.base import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.messages import AIMessage
from typing import List, Optional
from chat_utils import get_init_modules, SYSTEM_PROMPT, SYSTEM_PROMPT_LOOP, ContextInput, Answer, get_vectorDB_module
from langchain_core.documents.base import Document
from langchain_core.runnables import ConfigurableField
import qdrant_client.models as rest

class EurLexChat:
    def __init__(self, config: dict):
        self.config = config
        self.max_history_messages = self.config["max_history_messages"]
        self.vectorDB_class = self.config['vectorDB']['class']
        self.use_functions = (
            'use_context_function' in config["llm"] and
            config["llm"]["use_context_function"] and
            config["llm"]["class"] == "ChatOpenAI")

        self.embedder, self.llm, self.chatDB_class, self.retriever, retriever_chain = get_init_modules(
            config)


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
                "This function must not be called if you already have in the context the information to answer the user. ",
                args_schema=ContextInput
            )

            self.llm_with_functions = self.llm.bind(
                tools=[convert_to_openai_tool(GET_CONTEXT_TOOL)]
            )

            chain = ( 
                    self.prompt | 
                    RunnableLambda(self._resize_history) |
                    self.llm_with_functions
                    )
        else:
            chain = (
                    self.prompt | 
                    RunnableLambda(self._resize_history) |
                    self.llm
                    )

        self.chain_with_history = RunnableWithMessageHistory(
            chain,
            self.get_chat_history,
            input_messages_key="question",
            history_messages_key="history",
        )

        self.relevant_documents_pipeline = (retriever_chain | self._parse_documents)

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
            messages = [messages[0]] + messages[-(self.max_history_messages + 1):]
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
        elif self.config["chatDB"]["class"] == 'DynamoDBChatMessageHistory':
            table_name = kwargs["table_name"]
            session = boto3.Session(aws_access_key_id=kwargs["aws_access_key_id"],
                                    aws_secret_access_key=kwargs["aws_secret_access_key"],
                                    region_name='eu-west-1')
            return self.chatDB_class(session_id=session_id,
                                     table_name=table_name,
                                     boto3_session=session)
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
                'celex': doc.metadata["celex"],
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

    def get_ids_from_celexes(self, celex_list: List[str]):
        """
        Retrieve the IDs of the documents given their CELEX numbers.

        Args:
            celex_list (List[str]): A list of CELEX numbers.

        Returns:
            List[str]: A list of document IDs corresponding to the provided CELEX numbers
        """

        if self.vectorDB_class == 'Qdrant':
            scroll_filter = rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="celex",
                        match=rest.MatchAny(any=celex_list),
                    )
                ])
            offset = -1
            ids = []
            while not (offset is None and offset != -1):
                if offset == -1:
                    offset = None
                points, offset = self.retriever.vectorstore.client.scroll(
                    collection_name=self.retriever.vectorstore.collection_name,
                    limit=100,
                    offset=offset,
                    scroll_filter=scroll_filter,
                    with_payload=False
                )
                ids.extend([p.id for p in points])
        else:
            NotImplementedError(f"Not supported {self.vectorDB_class} vectorDB class")
        return ids

    def _get_qdrant_ids_filter(self, ids):
        """
        Returns a Qdrant filter to filter documents based on their IDs.

        This function acts as a workaround due to a hidden bug in Qdrant 
        that prevents correct filtering using CELEX numbers.

        Args:
            ids (List[str]): A list of document IDs.

        Returns:
            Qdrant filter: A Qdrant filter to filter documents based on their IDs.
        """

        filter = rest.Filter(
            must=[
                rest.HasIdCondition(has_id=ids),
            ],
        )

        return filter

    def get_relevant_docs(self, question: str, ids_list: Optional[List[str]] = None) -> List[dict]:
        """
        Retrieve relevant documents based on a given question.
        If ids_list is provided, the search is filtered by the given IDs.

        Args:
            question (str): The question for which relevant documents are retrieved.
            ids_list (Optional[List[str]]): A list of document IDs to filter the search results.

        Returns:
            List[dict]: A list of relevant documents.
        """
        if ids_list:
            search_kwargs = {k:v for k,v in self.retriever.search_kwargs.items()}
            if self.vectorDB_class == 'Qdrant':
                filter = self._get_qdrant_ids_filter(ids_list)
            else:
                raise ValueError(f'Celex filter not supported for {self.vectorDB_class}')

            search_kwargs.update({'filter': filter})
            docs = self.relevant_documents_pipeline.invoke(
                {'question': question},
                config={"configurable": {"search_kwargs": search_kwargs}})
        else:
            docs = self.relevant_documents_pipeline.invoke({'question': question})
        return docs

    def get_context(self, text: str, ids_list:Optional[List[str]]=None) -> str:
        """
        Retrieve context for a given text.
        If ids_list is provided, the search is filtered by the given IDs.

        Args:
            text (str): The text for which context is retrieved.
            ids_list (Optional[List[str]]): A list of document IDs to filter the search results.

        Returns:
            str: A formatted string containing the relevant documents texts.
        """

        docs = self.get_relevant_docs(text, ids_list=ids_list)
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

    def _resize_context(self, context_docs: List[dict]) -> List[dict]:
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

    def get_answer(self, 
                   session_id: str,
                   question: str,
                   context_docs: List[dict],
                   from_tool: bool = False,
                   ids_list: List[str] = None
                   ) -> Answer:
        """
        Get an answer to a question of a specific session, considering context documents and history messages.
        If ids_list is provided, any search for new context documents is filtered by the given IDs.

        Args:
            session_id (str): The session ID for which the answer is retrieved.
            question (str): The new user message.
            context_docs (List[dict]): A list of documents used as context to answer the user message.
            from_tool (bool, optional): Whether the question originates from a tool. Defaults to False.
            ids_list (Optional[List[str]]): A list of document IDs to filter the search results for new context documents.

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
            new_docs = self.get_relevant_docs(text, ids_list=ids_list)
            self._remove_last_messages(session_id=session_id, n=2)

            result = self.get_answer(
                session_id=session_id,
                question=question,
                context_docs=new_docs,
                from_tool=True,
                ids_list=ids_list
            )
            if result.status == 1:
                return Answer(answer=result.answer, new_documents=new_docs)
            else:
                return Answer(answer=result.answer)
        return Answer(answer=result.content)