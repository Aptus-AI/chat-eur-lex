import gradio as gr
from EurLexChat import EurLexChat
import random
import string
from config import CONFIG, UI_USER, UI_PWD
from consts import JUSTICE_CELEXES, POLLUTION_CELEXES
from enum import Enum
from copy import deepcopy

def generate_random_string(length):
    # Generate a random string of the specified length 
    # using letters and numbers
    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string

class ChatBot():
    def __init__(self, config) -> None:
        self.documents = []
        self.config = deepcopy(config)
        self.chat = EurLexChat(config=self.config)
    def __deepcopy__(self, _):
        return ChatBot(self.config)

class Versions(Enum):
    AKN='Akoma Ntoso'
    JUSTICE='Organisation of the legal system (1226) eurovoc'
    POLLUTION='Pollution (2524) eurovoc'
    BASIC='All eurovoc'


chat_init = EurLexChat(config=CONFIG)
justice_ids = chat_init.get_ids_from_celexes(JUSTICE_CELEXES)
pollution_ids = chat_init.get_ids_from_celexes(POLLUTION_CELEXES)


def reinit(version, bot):
    bot.documents = []
    config = deepcopy(CONFIG)
    if version == Versions.AKN.value:
        config['vectorDB']['kwargs']['collection_name'] += "-akn"
    bot.chat = EurLexChat(config=config)
    return clean_page(bot)

def remove_doc(btn, bot):
    bot.documents.pop(btn)
    new_accordions, new_texts = set_new_docs_ui(bot.documents)
    return [*new_accordions, *new_texts, bot]


def get_answer(message, history, session_id, celex_type, bot):
    s = session_id
    if celex_type == Versions.JUSTICE.value:
        ids_list = justice_ids
    elif celex_type == Versions.POLLUTION.value:
        ids_list = pollution_ids
    elif celex_type == Versions.BASIC.value or celex_type == Versions.AKN.value:
        ids_list = None
    else:
        raise ValueError(f'Wrong celex_type: {celex_type}')

    if len(history) == 0:
        bot.documents = []
        #docs.documents = chat.get_relevant_docs(question=message, ids_list=ids_list)
        s = generate_random_string(7)
    result = bot.chat.get_answer(s, message, bot.documents, ids_list=ids_list) 
    history.append((message, result.answer))
    if result.new_documents:
        bot.documents = result.new_documents
    accordions, list_texts = set_new_docs_ui(bot.documents)
    return ['', history, gr.Column(scale=1, visible=True), *accordions, *list_texts, s, bot]


def set_new_docs_ui(documents):
    new_accordions = []
    new_texts = []
    for i in range(len(accordions)):
        if i < len(documents):
            new_accordions.append(gr.update(accordions[i].elem_id, label=f"{documents[i]['celex']}: {documents[i]['text'][:40]}...", visible=True, open=False))
            new_texts.append(gr.update(list_texts[i].elem_id, value=f"{documents[i]['text']}...", visible=True))
        else:
            new_accordions.append(gr.update(accordions[i].elem_id, label="", visible=False))
            new_texts.append(gr.update(list_texts[i].elem_id, value="", visible=False))
    return new_accordions, new_texts


def clean_page(bot):
    bot.documents = []
    accordions, list_texts = set_new_docs_ui(bot.documents)
    return ["", [], None, *accordions, *list_texts, gr.Column(visible=False), bot]

list_texts = []
accordions = []
states = []
delete_buttons = []

if CONFIG['vectorDB'].get('rerank'):
    n_context_docs = CONFIG['vectorDB']['rerank']['kwargs']['top_n']
else:
    n_context_docs = CONFIG['vectorDB']['retriever_args']['search_kwargs']['k']

block = gr.Blocks()
with block:

    gr.Markdown("""
        <h1><center>Chat-EUR-Lex prototype - Alpha version</center></h1>
    """)
    bot = gr.State(value=ChatBot(CONFIG))
    state = gr.State(value=None)
    with gr.Row():
        with gr.Column(scale=3):
            drop_down = gr.Dropdown(label='Choose a version', choices=[attribute.value for attribute in Versions], value=Versions.BASIC)
            chatbot = gr.Chatbot()
            with gr.Row():
                message = gr.Textbox(scale=10,label='',placeholder='Write a message...', container=False)
                submit = gr.Button("Send message", scale=1)
                clear = gr.Button("Reset chat", scale=1)
            
        with gr.Column(scale=1, visible=False) as col:
            gr.Markdown("""<h3><center>Context documents</center></h3>""")
            for i in range(n_context_docs):
                with gr.Accordion(label="", elem_id=f'accordion_{i}', open=False) as acc:
                    list_texts.append(gr.Textbox("", elem_id=f'text_{i}', show_label=False, lines=10))
                    btn = gr.Button(f"Remove document")
                    delete_buttons.append(btn)
                    states.append(gr.State(i))
                accordions.append(acc)

    with gr.Row():
        with gr.Column(scale=3):
            gr.HTML("""""")
            gr.HTML("""<div>
                    <h3>Disclaimer</h3>
                    <p><a href="https://github.com/Aptus-AI/chat-eur-lex/">Chat-EUR-Lex prototype</a> is a limited risk AI system realized by the 
                    <a href="https://www.igsg.cnr.it/en/">Institute of Legal Informatics and Judicial Systems (IGSG-CNR)</a> and <a href="https://www.aptus.ai/">Aptus.AI</a>. 
                    The prototype is an AI chatbot, therefore you are interacting with a machine, not with a human person. The prototype uses OpenAI GPT-4 language model. </p>
                    
                    <p><a href="https://github.com/Aptus-AI/chat-eur-lex/">Chat-EUR-Lex project</a> is funded by the European Union within the framework of the NGI Search project under grant agreement No 101069364. 
                    Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or European Commission.
                    Contact us: <a href="mailto:chat-eur-lex@igsg.cnr.it">chat-eur-lex@igsg.cnr.it</a>.</p>
                    </div>""")
    
    drop_down.change(reinit, inputs=[drop_down, bot], outputs=[message, chatbot, state, *accordions, *list_texts, col])
    clear.click(clean_page, inputs=[bot], outputs=[message, chatbot, state, *accordions, *list_texts, col, bot])
    message.submit(get_answer, inputs=[message, chatbot, state, drop_down, bot], outputs=[message, chatbot, col, *accordions, *list_texts, state, bot])
    submit.click(get_answer, inputs=[message, chatbot, state, drop_down, bot], outputs=[message, chatbot, col, *accordions, *list_texts, state, bot])
    for i, b in enumerate(delete_buttons):
        b.click(remove_doc, inputs=[states[i],bot], outputs=[*accordions, *list_texts, bot])

block.launch(debug=True, auth=(UI_USER, UI_PWD))