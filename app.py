from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer

# Initialize the question-answering model
model_name = "deepset/roberta-base-squad2"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
reader = pipeline("question-answering", model=model, tokenizer=tokenizer)

# Function to read and process multiple CSV files
def read_and_process_files(filepaths):
    dataframes = [pd.read_csv(filepath) for filepath in filepaths]
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df

# List of file paths (you can add more file paths as needed)
filepaths = ['Item.csv', 'branches(1).csv']

# Read and combine the data
combined_df = read_and_process_files(filepaths)

# Function to create context from the entire DataFrame
def create_context(df):
    context = df.apply(lambda row: ' '.join(row.values.astype(str)), axis=1).str.cat(sep=' ')
    return context

# Create context for the API
context = create_context(combined_df)

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