from flask import Flask, jsonify
from transformers import pipeline
from time import time

app = Flask(__name__)

qna_model_holder = None

@app.route("/load")
def load_model():
    global qna_model_holder
    start = time()
    qna_model_holder = pipeline("question-answering", model='distilbert-base-cased-distilled-squad')
    end = time()

    print("\n=================================")
    print(qna_model_holder)
    print("=================================\n")

    response = {
        "start": start,
        "end": end,
        "total_time": str(round(((end - start)/60),2)) + " seconds",
        "msg": "Model loaded successfully!",
    }

    print("\n=================================")
    print(response)
    print("=================================\n")
    return jsonify(response)

@app.route("/qna")
def questionAnswer():
    context = r"""Extractive Question Answering is the task of extracting an answer from a text given a question. An example of a question answering dataset is the SQuAD dataset, which is entirely based on that task. If you would like to fine-tune a model on a SQuAD task, you may leverage the examples/pytorch/question-answering/run_squad.py script."""
    query = "What is a good example of a question answering dataset?"

    start = time()
    result = qna_model_holder(question=query, context=context)
    end = time()

    response = {
        "start": start,
        "end": end,
        "total_time": str(round(((end - start)/60),2)) + " seconds",
        "msg": "Answered successfully!",
        "result": result,
    }
    print(response)
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=false, host='0.0.0.0')
