# app.py
from flask import Flask, render_template, request
import pandas as pd
from module import linear_reg  # Replace with your ML module

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    target_variable = request.form['target']

    # Save the uploaded file
    file.save('uploaded_dataset.csv')  # Save as CSV for simplicity

    # Load the dataset
    dataset = pd.read_csv('uploaded_dataset.csv')

    # Analyze the dataset using your ML module
    result = linear_reg(dataset, target_variable)

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
