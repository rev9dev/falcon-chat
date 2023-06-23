import argparse
import os

import gradio as gr
from gradio import Blocks
from text_generation import Client

TITLE = """<h2 align="center">🚀 Falcon-Chat demo</h2>"""
USER_NAME = "User"
BOT_NAME = "Falcon"
DEFAULT_INSTRUCTIONS = f"""The following is a conversation between a highly knowledgeable and intelligent AI assistant, called Falcon, and a human user, called User. In the following interactions, User and Falcon will converse in natural language, and Falcon will answer User's questions. Falcon was built to be respectful, polite and inclusive. Falcon was built by the Technology Innovation Institute in Abu Dhabi. Falcon will never decline to answer a question, and always attempts to give an answer that User would be satisfied with. It knows a lot, and always tells the truth. The conversation begins.
"""
RETRY_COMMAND = "/retry"
STOP_STR = f"\n{USER_NAME}:"
STOP_SUSPECT_LIST = [":", "\n", "User"]

INFERENCE_ENDPOINT = os.environ.get("INFERENCE_ENDPOINT")
INFERENCE_AUTH = os.environ.get("INFERENCE_AUTH")


def chat_accordion():
    with gr.Accordion("Parameters", open=False):
        temperature = gr.Slider(
            minimum=0.1,
            maximum=2.0,
            value=0.8,
            step=0.1,
            interactive=True,
            label="Temperature",
        )
        top_p = gr.Slider(
            minimum=0.1,
            maximum=0.99,
            value=0.9,
            step=0.01,
            interactive=True,
            label="p (nucleus sampling)",
        )
    return temperature, top_p


def format_chat_prompt(message: str, chat_history, instructions: str) -> str:
    instructions = instructions.strip(" ").strip("\n")
    prompt = instructions
    for turn in chat_history:
        user_message, bot_message = turn
        prompt = f"{prompt}\n{USER_NAME}: {user_message}\n{BOT_NAME}: {bot_message}"
    prompt = f"{prompt}\n{USER_NAME}: {message}\n{BOT_NAME}:"
    return prompt


def chat(client: Client):
    chatbot = gr.Chatbot(elem_id="chatbot")  # Move the Chatbot outside the container

    inputs = gr.Textbox(
        placeholder=f"Send a message",
        max_lines=3,
        show_label=False,
    )  # Move the Textbox outside the container

    with gr.Row():
        with gr.Column(size="20%"):
            with gr.Accordion("Navigation", open=True):
                retry_button = gr.Button("♻️ Retry last turn")
                delete_turn_button = gr.Button("🧽 Delete last turn")
                clear_chat_button = gr.Button("✨ Delete all history")

        with gr.Column(size="80%"):
            inputs  # Place the Textbox in the remaining space

    with gr.Row(elem_id="param_container"):
        with gr.Column():
            temperature, top_p = chat_accordion()
        with gr.Column():
            with gr.Accordion("Instructions", open=False):
                instructions = gr.Textbox(
                    placeholder="LLM instructions",
                    value=DEFAULT_INSTRUCTIONS,
                    lines=10,
                    interactive=True,
                    label="Instructions",
                    max_lines=16,
                    show_label=False,
                )

    def run_chat(message: str, chat_history, instructions: str, temperature: float, top_p: float):
        if not message or (message == RETRY_COMMAND and len(chat_history) == 0):
            yield chat_history
            return

        if message == RETRY_COMMAND and chat_history:
            prev_turn = chat_history.pop(-1)
            user_message, _ = prev_turn
            message = user_message

        prompt = format_chat_prompt(message, chat_history, instructions)
        chat_history = chat_history + [[message, ""]]
        stream = client.generate_stream(
            prompt,
            do_sample=True,
            max_new_tokens=1024,
            stop_sequences=[STOP_STR, "<|endoftext|>"],
            temperature=temperature,
            top_p=top_p,
        )
        acc_text = ""
        for idx, response in enumerate(stream):
            text_token = response.token.text

            if response.details:
                return

            if text_token in STOP_SUSPECT_LIST:
                acc_text += text_token
                continue

            if idx == 0 and text_token.startswith(" "):
                text_token = text_token[1:]

            acc_text += text_token
            last_turn = list(chat_history.pop(-1))
            last_turn[-1] += acc_text
            chat_history = chat_history + [last_turn]
            yield chat_history
            acc_text = ""

    def delete_last_turn(chat_history):
        if chat_history:
            chat_history.pop(-1)
        return {chatbot: gr.update(value=chat_history)}

    def run_retry(message: str, chat_history, instructions: str, temperature: float, top_p: float):
        yield from run_chat(RETRY_COMMAND, chat_history, instructions, temperature, top_p)

    def clear_chat():
        return []

    inputs.submit(
        run_chat,
        [inputs, chatbot, instructions, temperature, top_p],
        outputs=[chatbot],
        show_progress=False,
    )
    inputs.submit(lambda: "", inputs=None, outputs=inputs)
    delete_turn_button.click(delete_last_turn, inputs=[chatbot], outputs=[chatbot])
    retry_button.click(
        run_retry,
        [inputs, chatbot, instructions, temperature, top_p],
        outputs=[chatbot],
        show_progress=False,
    )
    clear_chat_button.click(clear_chat, [], chatbot)


def get_demo(client: Client):
    demo: Blocks
    with gr.Blocks(
            # css=None
            # css="""#chat_container {width: 700px; margin-left: auto; margin-right: auto;}
            #        #button_container {width: 700px; margin-left: auto; margin-right: auto;}
            #        #param_container {width: 700px; margin-left: auto; margin-right: auto;}"""
            css="""#chatbot {
    width: 750px !important;
    margin: 0 auto !important;
}"""
    ) as demo:
        gr.HTML(TITLE)
        chat(client)

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Playground Demo")
    parser.add_argument(
        "--addr",
        type=str,
        required=False,
        default=INFERENCE_ENDPOINT,
    )
    args = parser.parse_args()
    client = Client(args.addr, headers={"Authorization": f"Basic {INFERENCE_AUTH}"})
    demo = get_demo(client)
    demo.queue(max_size=128, concurrency_count=16)
    demo.launch(share=True)
