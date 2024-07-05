import gradio as gr
from EurLexChat import EurLexChat
import random
import string
from config import CONFIG, UI_USER, UI_PWD
from consts import JUSTICE_CELEXES, POLLUTION_CELEXES

def generate_random_string(length):
    # Generate a random string of the specified length 
    # using letters and numbers
    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string

class Documents():
    def __init__(self) -> None:
        self.documents = []


chat = EurLexChat(config=CONFIG)

justice_ids = chat.get_ids_from_celexes(JUSTICE_CELEXES)
pollution_ids = chat.get_ids_from_celexes(POLLUTION_CELEXES)

docs = Documents()


def remove_doc(btn):
    docs.documents.pop(btn)
    new_accordions, new_texts = set_new_docs_ui(docs.documents)
    return [*new_accordions, *new_texts]


def get_answer(message, history, session_id, celex_type):
    s = session_id

    if celex_type == 'justice':
        ids_list = justice_ids
    elif celex_type == 'pollution':
        ids_list = pollution_ids
    elif celex_type is None:
        ids_list = []
    else:
        raise ValueError(f'Wrong celex_type: {celex_type}')

    if len(history) == 0:
        docs.documents = []
        #docs.documents = chat.get_relevant_docs(question=message, ids_list=ids_list)
        s = generate_random_string(7)
    result = chat.get_answer(s, message, docs.documents, ids_list=ids_list) 
    history.append((message, result.answer))
    if result.new_documents:
        docs.documents = result.new_documents
    accordions, list_texts = set_new_docs_ui(docs.documents)
    return ['', history, gr.Column(scale=1, visible=True), *accordions, *list_texts, s]


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


def clean_page():
    docs.documents = []
    accordions, list_texts = set_new_docs_ui(docs.documents)
    return ["", [], None, *accordions, *list_texts]

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
    state = gr.State(value=None)
    with gr.Row():
        with gr.Column(scale=3):
            radio = gr.Radio(label='Choose a topic', choices=['justice','pollution'])
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
    
    clear.click(clean_page, outputs=[message, chatbot, state, *accordions, *list_texts])
    message.submit(get_answer, inputs=[message, chatbot, state, radio], outputs=[message, chatbot, col, *accordions, *list_texts, state])
    submit.click(get_answer, inputs=[message, chatbot, state, radio], outputs=[message, chatbot, col, *accordions, *list_texts, state])
    for i, b in enumerate(delete_buttons):
        b.click(remove_doc, inputs=states[i], outputs=[*accordions, *list_texts])

block.launch(debug=True, auth=(UI_USER, UI_PWD))