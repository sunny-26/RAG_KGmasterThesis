from langchain.callbacks.manager import CallbackManager
from langchain_community.llms import LlamaCpp
from huggingface_hub import hf_hub_download
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from ExtraFunction import setup_llama_model
from ExtraFunctionKGCommunication import get_from_kg

from ExtraFunctionsRAG import create_sub_graph

from ExtraFunctionForEntitiesExtraction import extract_json,put_data_into_query_template
import json


from llama_index.core.indices.service_context import ServiceContext
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.indices.knowledge_graph import KnowledgeGraphIndex
from llama_index.core.graph_stores import SimpleGraphStore

from flask import Flask, request, jsonify

app = Flask(__name__)

def chat_with_llm(question):

    # Step 1: Retrieve variable and its value
    GENERATIVE_AI_MODEL_REPO = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
    GENERATIVE_AI_MODEL_FILE = "./mistral-7b-instruct-v0.2.Q4_K_M.gguf"

    model_path = hf_hub_download(
        repo_id=GENERATIVE_AI_MODEL_REPO,
        filename=GENERATIVE_AI_MODEL_FILE
    )

    f_path = r"C:\Users\daria\PycharmProjects\SendRequest\venv\Resources\json.gbnf"


    extraInstruction = "If you do not know, say, that you do not know."
    parameters_for_query = extract_json(model_path, f_path, question)
    print(parameters_for_query)

    # Step 2: send request to Knowledge graph

    # 2. Create query using template
    sparql_query = put_data_into_query_template(json.loads(parameters_for_query))

    # 3. Sneding request
    sparql_endpoint = "https://triplestore1.informatik.tu-chemnitz.de/sparql/"
    result = get_from_kg(sparql_query, sparql_endpoint)


    # Step 3: creating sub graph
    node_FromGraph_tups = []
    create_sub_graph(node_FromGraph_tups, result)

    # Step 4: RAG implamentation

    # Initialize LLM
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    GENERATIVE_AI_MODEL_REPO = "TheBloke/stablelm-zephyr-3b-GGUF"
    GENERATIVE_AI_MODEL_FILE = "./stablelm-zephyr-3b.Q4_K_M.gguf"

    llm_model = setup_llama_model(GENERATIVE_AI_MODEL_REPO, GENERATIVE_AI_MODEL_FILE,
                                  callback_handler=StreamingStdOutCallbackHandler())

    # initialize Graph to connect with llm
    service_context = ServiceContext.from_defaults(llm=llm_model, embed_model='local')

    graph_store = SimpleGraphStore()
    for tup in node_FromGraph_tups:
        subject, predicate, obj = tup
        graph_store.upsert_triplet(subject, predicate, obj)

    storage_context = StorageContext.from_defaults(graph_store=graph_store)

    index = KnowledgeGraphIndex(
        [],
        service_context=service_context,
        storage_context=storage_context,
    )

    query_engine = index.as_query_engine(
        include_text=False, response_mode="tree_summarize"
    )

    response = query_engine.query(
        question + extraInstruction,

    )
    return str(response)


@app.route('/chat', methods=['POST'])
def ask():
    input_question = request.json.get('question')
    if input_question:
        answer = chat_with_llm(input_question)
        return answer
    else:
        return 'No question provided', 400

if __name__ == '__main__':
    app.run(debug=True)


