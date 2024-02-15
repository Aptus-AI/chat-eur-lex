import gradio as gr
from EurLexChat import EurLexChat
import yaml
import random
import string

def generate_random_string(length):
    # Generate a random string of the specified length 
    # using letters and numbers
    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string

class Documents():
    def __init__(self) -> None:
        self.documents = []

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

chat = EurLexChat(config=config)
docs = Documents()


def remove_doc(btn):
    docs.documents.pop(btn)
    new_accordions, new_texts = set_new_docs_ui(docs.documents)
    return [*new_accordions, *new_texts]


def get_answer(message, history, session_id):
    if len(history) == 0:
        docs.documents = chat.get_relevant_docs(question=message)
    result = chat.get_answer(session_id, message, docs.documents)
    history.append((message, result.answer))
    if result.new_documents:
        docs.documents = result.new_documents
    accordions, list_texts = set_new_docs_ui(docs.documents)
    return ['', history, gr.Column(scale=1, visible=True), *accordions, *list_texts]


def set_new_docs_ui(documents):
    new_accordions = []
    new_texts = []
    for i in range(len(accordions)):
        if i < len(documents):
            new_accordions.append(gr.update(accordions[i].elem_id, label=f"{documents[i]['text'][:45]}...", visible=True, open=False))
            new_texts.append(gr.update(list_texts[i].elem_id, value=f"{documents[i]['text']}...", visible=True))
        else:
            new_accordions.append(gr.update(accordions[i].elem_id, label="", visible=False))
            new_texts.append(gr.update(list_texts[i].elem_id, value="", visible=False))
    return new_accordions, new_texts


list_texts = []
accordions = []
states = []
delete_buttons = []

block = gr.Blocks()
with block:

    gr.Markdown("""
        <h1><center>Chat with Eur-Lex</center></h1>
    """)
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot()
            with gr.Row():
                message = gr.Textbox(scale=10)
                submit = gr.Button("Send", scale=1)
            
        with gr.Column(scale=1, visible=False) as col:
            gr.Markdown("""<h3><center>Context documents</center></h3>""")
            for i in range(config['vectorDB']['retriever_args']['search_kwargs']['k']):
                with gr.Accordion(label="", elem_id=f'accordion_{i}', open=False) as acc:
                    list_texts.append(gr.Textbox("", elem_id=f'text_{i}', show_label=False, lines=10))
                    btn = gr.Button(f"Remove document")
                    delete_buttons.append(btn)
                    states.append(gr.State(i))
                accordions.append(acc)
    
    state = gr.State(value=generate_random_string(7))
    message.submit(get_answer, inputs=[message, chatbot, state], outputs=[message, chatbot, col, *accordions, *list_texts])
    submit.click(get_answer, inputs=[message, chatbot, state], outputs=[message, chatbot, col, *accordions, *list_texts])
    for i, b in enumerate(delete_buttons):
        b.click(remove_doc, inputs=states[i], outputs=[*accordions, *list_texts ])

block.launch(debug=True)