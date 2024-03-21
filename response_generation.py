# here the function to construct the prompt and send it
# to recieve the trustworthy answer from LLM are stored

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages import AIMessage, HumanMessage
from support_functions import local_model
def adapt_extracted_data(kg): # make data more readable
    dictionary = {}
    context = ""
    # Organize tuples into the dictionary
    for url, key, value in kg:
        if url not in dictionary:
            dictionary[url] = {}
        dictionary[url][key] = value

    # Construct context from min-KG
    for url, attributes in dictionary.items():
        context += "URL: " + url + "\n"
        for key, value in attributes.items():
            context += key.capitalize() + ": " + value + "\n"
        context += "\n"
    return context


def construct_system_prompt_for_rag(rag_context):

    role="You are an expert in answering question based on the context only."
    instructions1="Use the following context to answer the question:"+rag_context
    instructions2="Tell, that you do not know, if the context is not enough to generate a trustworthy answer."

    system_prompt=role+instructions1+instructions2

    return system_prompt




def execute_rag(model_path, question,context, n_gpu_layers=64, n_ctx=4096,n_batch=3):
    # initialize

    llm = local_model( model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx, n_batch=n_batch)
    messages = [
        SystemMessage(
            content=context
        ),
        HumanMessage(content=question),
    ]

    response = llm.invoke(input=messages)


    return response