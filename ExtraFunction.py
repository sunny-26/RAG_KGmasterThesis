
from langchain.callbacks.manager import CallbackManager
from langchain_community.llms import LlamaCpp
from huggingface_hub import hf_hub_download
from keybert import KeyBERT
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.messages import HumanMessage, SystemMessage
import json

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
        n_batch=10,
        max_tokens=400,
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

#function to retrieve only required from query data
def extract_keywords(question):
    kw_model = KeyBERT(model='all-mpnet-base-v2')

    keywords = kw_model.extract_keywords(question, keyphrase_ngram_range=(1, 3),highlight=False,top_n=10)

    keywords_list = list(dict(keywords).keys())
    # print(keywords_list)
    return keywords_list

def extract_synonyms_based_on_graph_predicates(keywords_list,properties):
    similarities_by_word = {}
    nlp = spacy.load("en_core_web_sm")
    for word in keywords_list:
        word_vector = nlp(word).vector
        similarities = [(prop, cosine_similarity([word_vector], [nlp(prop).vector])[0][0]) for prop in properties]
        similarities_by_word[word] = similarities
    predicates_to_construct_context = []
    # Display similarities for each word
    for word, similarities in similarities_by_word.items():

        for prop, score in similarities:
            if (score > 0.7):
                predicates_to_construct_context.append(prop)
    return predicates_to_construct_context

#experiment with main subject (or is it better to do with the publication or any other keyword)
def create_subgraph(node_FromGraph_tups,json_data):
  for binding in json_data['results']['bindings']:
    for var, value in binding.items():
      if value['type'] == 'literal':
        tup = ("Selected publication", var, value['value'])
      elif value['type'] == 'uri':
        tup = ("Selected publication", var, f"URI: {value['value']}")
      elif value['type'] == 'typed-literal':
        tup = ("Selected publication", var, f"{value['datatype']}: {value['value']}")
      else:
        tup = ("Selected publication", var, str(value))
      node_FromGraph_tups.append(tup)



def extract_data(knowledge_graph,predicates):
    matching_tuples = []
    for triple in knowledge_graph:
        if triple[1] in predicates:
            matching_tuples.append(triple)
    return matching_tuples


def construct_mini_graph(node_FromGraph_tups,list_of_possible_predicates):
    predicates = []
    # add titles if additionally to title
    for triple in node_FromGraph_tups:
        predicates.append(triple[1])
    if "title" not in list_of_possible_predicates and "title" in predicates:
        list_of_possible_predicates.append("title")
    rag_context = extract_data(node_FromGraph_tups, list_of_possible_predicates)
    return rag_context


def construct_system_prompt_for_rag(rag_context):
    #question is a part of human prompt
    role="You are an expert in answering question based on the context only"
    instructions1="Use the following context to answer the question:"+rag_context
    instructions2="Tell, that you do not know if the conxtext is not enough to provide an answer"
    examples="""
    """
    example="Please, use the following examples in order to provide a truthful and relevant answer:"+examples

    system_prompt=role+instructions1+instructions2+example

    return system_prompt


#rag request

async def RAG_query(context, question,model):

    response = await model.invoke([
         SystemMessage(
            context
        ),
         HumanMessage(question),
    ])


def extract_json_form_LLM_api(text):
    json_objects = []
    start_idx = 0

    while True:
        start_idx = text.find('{', start_idx)
        if start_idx == -1:
            break

        # end index of the json object
        end_idx = text.find('}', start_idx)
        if end_idx == -1:
            break


        json_str = text[start_idx:end_idx + 1]

        try:
            # Try to parse the extracted string as JSON
            json_obj = json.loads(json_str)
            json_objects.append(json_obj)
        except json.JSONDecodeError:
            # If parsing fails, continue searching for next JSON object
            pass

        # Move the start index to the end of the current JSON object
        start_idx = end_idx + 1

    return json_objects
