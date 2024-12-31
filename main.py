import os
import json
import PySimpleGUI as sg
import openai
import tiktoken

# -----------------------------------------
# CONFIGURATION
# -----------------------------------------
OPENAI_API_KEY = "OPEN AI KEY"  # <-- Put your key here or handle via env var
MODEL_NAME = "o1-mini-2024-09-12"   # e.g., "o1-mini-2024-09-12", "gpt-4o-mini", etc.
CHUNK_SIZE = 3000                   # approximate max tokens per chunk
HISTORY_JSON_PATH = "conversation_history.json"

# Set API key
openai.api_key = OPENAI_API_KEY


def count_tokens(text: str, model_name: str = MODEL_NAME) -> int:
    """
    Uses tiktoken to approximate the number of tokens in a string
    with the given model's encoding.
    """
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(text))


def chunk_text(text: str, max_tokens: int = CHUNK_SIZE) -> list[str]:
    """
    Splits text into chunks that are each up to `max_tokens` tokens.
    Returns a list of chunk strings.
    """
    paragraphs = text.split('\n')
    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        candidate = (current_chunk + "\n" + paragraph).strip()
        if count_tokens(candidate) > max_tokens and current_chunk:
            # push the existing chunk
            chunks.append(current_chunk)
            current_chunk = paragraph
        else:
            if not current_chunk:
                current_chunk = paragraph
            else:
                current_chunk += "\n" + paragraph

    # Append any leftover text
    if current_chunk.strip():
        chunks.append(current_chunk)

    return chunks


def load_or_init_history(filepath: str):
    """
    Load conversation history from a local JSON file if exists;
    otherwise return an empty structure.
    """
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    else:
        return []


def save_history(filepath: str, history):
    """
    Save conversation history into a local JSON file.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def add_developer_message(history, content):
    """
    Add a developer (system-like) message to the conversation history array.
    Many newer models do not support the 'system' role.
    """
    history.append({"role": "developer", "content": content})


def add_user_message(history, content):
    """
    Add a user message to the conversation history array.
    """
    history.append({"role": "user", "content": content})


def add_assistant_message(history, content):
    """
    Add an assistant message (the model response) to the conversation history array.
    """
    history.append({"role": "assistant", "content": content})


def send_chat_completion(history):
    """
    Send the conversation history to the Chat Completion endpoint for the selected model.
    NOTE: For o1-mini, some parameters (like temperature) might not be supported
    or must be set to default = 1.
    """
    # The minimal set of parameters for an o1-mini call
    response = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=history,
        temperature=1  # Must be 1 for o1 models that don't allow changes
    )
    return response.choices[0].message["content"]


def main():
    # Load existing conversation or start fresh
    conversation_history = load_or_init_history(HISTORY_JSON_PATH)

    # If conversation is empty, add a "developer" message with instructions
    if not conversation_history:
        initial_instructions = (
            "You are a helpful code assistant that can review, debug, and discuss code. "
            "You have a limited context window, so keep your answers concise. "
            "Your role is 'developer', which is like a system role for some models."
        )
        add_developer_message(conversation_history, initial_instructions)
        save_history(HISTORY_JSON_PATH, conversation_history)

    # ----------------------------------
    # PySimpleGUI Layout
    # ----------------------------------
    sg.theme("SystemDefault")

    layout = [
        [sg.Text("Personal Developer Agent", font=("Helvetica", 14, "bold"))],
        [sg.Text("Loaded Files:")],
        [sg.Multiline(size=(80, 5), key="-FILES-", disabled=True)],
        [
            sg.Button("Select Files", key="-SELECT_FILES-"),
            sg.Button("Clear Files", key="-CLEAR_FILES-"),
            sg.Button("Clear Conversation", key="-CLEAR_CONVO-"),
            sg.Button("Save Output", key="-SAVE_OUTPUT-")
        ],
        [sg.Text("Your Prompt:")],
        [sg.Multiline(size=(80, 5), key="-PROMPT-")],
        [sg.Button("Send Prompt", key="-SEND-"), sg.Button("Exit", key="-EXIT-")],
        [sg.Text("Assistant Response:")],
        [sg.Multiline(size=(80, 15), key="-RESPONSE-", disabled=True)]
    ]

    window = sg.Window("o1-mini Developer Agent", layout, resizable=True)
    loaded_files = []

    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, "-EXIT-"):
            break

        elif event == "-SELECT_FILES-":
            file_paths = sg.popup_get_file(
                "Select code files",
                multiple_files=True,
                file_types=(
                    ("Code Files", "*.cs *.ts *.js *.py *.cpp *.c *.h *.java *.txt"),
                    ("All Files", "*.*")
                )
            )
            if file_paths:
                # On Windows, file_paths may be a string separated by ';'
                if isinstance(file_paths, str):
                    file_paths = file_paths.split(";")

                for fp in file_paths:
                    fp = fp.strip()
                    if fp and os.path.isfile(fp):
                        loaded_files.append(fp)
                window["-FILES-"].update("\n".join(loaded_files))

        elif event == "-CLEAR_FILES-":
            loaded_files = []
            window["-FILES-"].update("")

        elif event == "-CLEAR_CONVO-":
            conversation_history = []
            save_history(HISTORY_JSON_PATH, conversation_history)
            window["-RESPONSE-"].update("")
            sg.popup("Conversation cleared. Next prompt will start fresh developer instructions.")

        elif event == "-SEND-":
            user_prompt = values["-PROMPT-"].strip()
            if not user_prompt and not loaded_files:
                sg.popup("Please either enter a prompt or load files to send.")
                continue

            # 1. If there are newly selected files, chunk & add them
            for fp in loaded_files:
                with open(fp, "r", encoding="utf-8") as f:
                    code_content = f.read()
                code_chunks = chunk_text(code_content, CHUNK_SIZE)
                for idx, chunk in enumerate(code_chunks, start=1):
                    chunk_msg = f"[FILE: {os.path.basename(fp)}, PART {idx}]\n{chunk}"
                    add_user_message(conversation_history, chunk_msg)

            # Clear the loaded file list to avoid re-sending the same code repeatedly
            loaded_files = []
            window["-FILES-"].update("")

            # 2. If user typed an actual prompt, add it
            if user_prompt:
                add_user_message(conversation_history, user_prompt)

            # 3. Send conversation to the model
            try:
                answer = send_chat_completion(conversation_history)
                add_assistant_message(conversation_history, answer)
            except Exception as e:
                answer = f"Error: {str(e)}"

            # 4. Show the assistant response in the UI
            window["-RESPONSE-"].update(answer)

            # 5. Save the new conversation state
            save_history(HISTORY_JSON_PATH, conversation_history)
            # 6. Clear the user prompt
            window["-PROMPT-"].update("")

        elif event == "-SAVE_OUTPUT-":
            save_path = sg.popup_get_file(
                "Save conversation output as:",
                save_as=True,
                default_extension=".txt",
                file_types=(("Text Files", "*.txt"), ("All Files", "*.*"))
            )
            if save_path:
                try:
                    with open(save_path, "w", encoding="utf-8") as f:
                        for msg in conversation_history:
                            role = msg["role"].upper()
                            content = msg["content"]
                            f.write(f"{role}:\n{content}\n\n")
                    sg.popup(f"Conversation saved to {save_path}")
                except Exception as e:
                    sg.popup(f"Failed to save file: {str(e)}")

    window.close()


if __name__ == "__main__":
    main()
