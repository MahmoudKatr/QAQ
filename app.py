from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import fitz  # PyMuPDF
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer

# Initialize the question-answering model
model_name = "deepset/roberta-base-squad2"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
reader = pipeline("question-answering", model=model, tokenizer=tokenizer)

# Function to read and process multiple PDF files
def read_and_process_files(filepaths):
    texts = []
    for filepath in filepaths:
        with fitz.open(filepath) as doc:
            for page in doc:
                texts.append(page.get_text())
    combined_text = ' '.join(texts)
    return combined_text

# List of file paths (you can add more file paths as needed)
filepaths = ['Combined_Item_Branch_Report.pdf']

# Read and combine the data
context = read_and_process_files(filepaths)

# Function to get response using the question-answering model
def get_response(question):
    outputs = reader(question=question, context=context)
    return outputs['answer']

# Initialize the Flask app
app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["GET"])
def predict():
    try:
        text = request.args.get("message")

        if text:
            response = get_response(text)
            message = {"answer": response}
            return jsonify(message)
        else:
            return jsonify({"error": "No message provided"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)