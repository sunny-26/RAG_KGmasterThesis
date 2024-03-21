# the functions for creating relevant context to augument the LLM response are stored
from keybert import KeyBERT
import spacy
from sklearn.metrics.pairwise import cosine_similarity

#regon Create a graph from data, retrieved from main graph
def create_sub_graph(node_FromGraph_tups,json_data):
  for binding in json_data['results']['bindings']:
    for var, value in binding.items():
      if value['type'] == 'literal':
        tup = (binding['publication']['value'], var, value['value'])
      elif value['type'] == 'uri':
        tup = (binding['publication']['value'], var, f"URI: {value['value']}")
      elif value['type'] == 'typed-literal':
        tup = (binding['publication']['value'], var, f"{value['datatype']}: {value['value']}")
      else:
        tup = (binding['publication']['value'], var, str(value))
      node_FromGraph_tups.append(tup)

#endregion

#region Retrieve only relevant data to cunstruct the context for LLM prompting

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


def extract_data(knowledge_graph,predicates): # extract only tuples where predicate matches keywords
    matching_tuples = []
    for triple in knowledge_graph:
        if triple[1] in predicates:
            matching_tuples.append(triple)
    return matching_tuples

# End title to the extracted KG, to provide a better context
def construct_mini_graph(node_FromGraph_tups,list_of_possible_predicates):
    predicates = []
    # add titles if additionally to title
    for triple in node_FromGraph_tups:
        predicates.append(triple[1])
    if "title" not in list_of_possible_predicates and "title" in predicates:
        list_of_possible_predicates.append("title")
    rag_context = extract_data(node_FromGraph_tups, list_of_possible_predicates)
    return rag_context



#endregion