# set the communication with the LLM via api, to access via other UIs (local server establishment)
from flask import Flask, request, jsonify

from rag_pipeline import RAG_Pipeline



app = Flask(__name__)


@app.route('/chat', methods=['POST'])
def ask():
    input_question = request.json.get('question')
    if input_question:
        answer = RAG_Pipeline(input_question)
        return answer
    else:
        return 'No question provided', 400




if __name__ == '__main__':
    app.run(debug=True)




