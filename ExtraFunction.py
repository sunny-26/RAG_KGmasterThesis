
from langchain.callbacks.manager import CallbackManager
from langchain_community.llms import LlamaCpp
from huggingface_hub import hf_hub_download


#Set LLM
def setup_llama_model(model_repo, model_file, callback_handler=None):
    # Callbacks support token-wise streaming
    callback_manager = CallbackManager([callback_handler])

    model_path = hf_hub_download(
        repo_id=model_repo,
        filename=model_file
    )

    llm = LlamaCpp(
        model_path=model_path,
        temperature=0.0,

        n_ctx=4096,
        #top_p=1,
        callback_manager=callback_manager,
        verbose=True  # Verbose is required to pass to the callback manager
    )
    return llm

def read_file_to_string(file_path):
    try:
        with open(file_path, 'r') as file:
            file_content = file.read()
        return file_content
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return None

