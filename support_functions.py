# here all the functions which are not directly connected to RAG are stored
import json
from langchain_community.llms import LlamaCpp
import os
from openai import OpenAI
from dotenv import load_dotenv
#function to extract json from the text
def extract_json_from_llm_response(entities):
    start_brace = 0
    entities_json = []
    isCompleted=False

    while True:
        # start of the json object
        start_brace = entities.find('{', start_brace)
        if start_brace == -1:
            break

        # end of the json object
        end_brace = entities.find('}', start_brace)
        if end_brace == -1:
            break

        # Extract the substring containing the JSON object
        json_str = entities[start_brace:end_brace + 1]

        try:
            # Parsing string as json object
            json_obj = json.loads(json_str)
            entities_json.append(json_obj)
        except json.JSONDecodeError:
            pass
        # end of construction
        start_brace = end_brace + 1

    return entities_json


def api_model():
    #load_dotenv()
    client = OpenAI(
        api_key=os.getenv('API_KEY'),
        base_url=os.getenv('BASE_URL')
    )
    return client


def rag_api_model(client, messages):
    #load_dotenv()
    completion = client.chat.completions.create(
        model=os.getenv('MODEL_API'),
        #seed=12345,
        max_tokens=300,
        messages=messages

    )
    return completion.choices[0].message.content



def local_model(model_path, n_gpu_layers=64, n_ctx=4096,n_batch=3):
    llm = LlamaCpp(model_path=model_path,
                   n_gpu_layers=n_gpu_layers,
                   n_ctx=n_ctx,n_batch=n_batch)
    return llm


