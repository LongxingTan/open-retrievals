import os
import shutil
from functools import partial

import gradio as gr

from retrievals.pipelines import KnowledgeCenter, ModelCenter


class CFG:
    llm_model_name_or_path = "../models"
    embed_model_name_or_path = "BAAI/bge-large-zh-v1.5"
    vector_db_path = "./database/chroma"


def upload_file(file, file_list, knowledge_center):
    """User upload"""
    if not os.path.exists("docs"):
        os.mkdir("docs")
    filename = os.path.basename(file.name)
    shutil.move(file.name, "docs/" + filename)
    file_list.insert(0, filename)
    knowledge_center.add_document("docs/" + filename)
    return gr.Dropdown.update(choices=file_list, value=filename)


def clear_session():
    return "", None, ""


def main():
    model_center = ModelCenter(CFG)
    knowledge_center = KnowledgeCenter(CFG)

    block = gr.Blocks()
    with block as demo:
        with gr.Row(equal_height=True):
            with gr.Column(scale=15):
                gr.Markdown("""<h1><center>RAG powered by Open-retrievals</center></h1>""")
            # gr.Image(value=LOGO_PATH, scale=1, min_width=10,show_label=False, show_download_button=False)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Dropdown(
                    ["text2vec-base", "bge-base"],
                    label="embedding model",
                    value="text2vec-base",
                )
                gr.Dropdown(["InternLM"], label="large language model", value="InternLM")
                gr.Slider(1, 20, value=3, step=1, label="retrieval topk doc", interactive=True)
                gr.Button("Load knowledge")
                file = gr.File(
                    label="Upload to knowledge center",
                    visible=True,
                    file_types=[".txt", ".md", ".docx", ".pdf"],
                )

            with gr.Column(scale=4):
                with gr.Row():
                    chatbot = gr.Chatbot(height=400, show_copy_button=True)
                with gr.Row():
                    # input box for prompt
                    message = gr.Textbox(label="Please input question")
                with gr.Row():
                    # clear button
                    gr.ClearButton(components=[chatbot], value="Clear history")
                    # submit and chat button
                    db_wo_his_btn = gr.Button("Generate chat")
                with gr.Row():
                    # retrieval context
                    search = gr.Textbox(label="Reference", max_lines=10)

            # click upload, add knowledge by knowledge center
            file.upload(
                partial(upload_file, knowledge_center=knowledge_center),
                inputs=[file],
                outputs=None,
            )

            # click chat
            db_wo_his_btn.click(
                model_center.qa_chain_self_answer,
                inputs=[message, chatbot],
                outputs=[message, chatbot, search],
            )
            # enter
            message.submit(
                model_center.qa_chain_self_answer,
                inputs=[message, chatbot],
                outputs=[message, chatbot, search],
            )

        gr.Markdown(
            """Warning：<br>
            1. It's just a demo page from open-retrievals <br>
            """
        )
    # threads to consume the request
    gr.close_all()

    # server
    demo.queue()
    demo.launch()


if __name__ == "__main__":
    main()
