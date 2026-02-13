from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
import time
import hashlib
import base64
import distance
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/')
def home():  
    return render_template('index.html', output=None, time_taken=None, file_download=None)

@app.route('/uploads')
def list_uploads():
    files = os.listdir(UPLOAD_FOLDER)
    files = [f for f in files if os.path.isfile(os.path.join(UPLOAD_FOLDER, f))]
    return render_template('uploads.html', files=files)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', output="⚠️ No file uploaded", time_taken=None, file_download=None)
    
    file = request.files['file']
    output_filename = request.form.get('filename', 'deduped_output.csv').strip()
    if file.filename == '':
        return render_template('index.html', output="⚠️ No selected file", time_taken=None, file_download=None)
    
    if not output_filename.endswith('.csv'):
        output_filename += '.csv'
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    output_path = os.path.join(UPLOAD_FOLDER, output_filename)
    file.save(file_path)

    # Run deduplication logic
    result_text, output_path, time_taken = run_deduplication(file_path, output_path)

    return render_template(
        'index.html',
        output=result_text,
        time_taken=time_taken,
        file_download=output_path
    )


def run_deduplication(file_path, output_path):
    start_h = time.time()
    data = pd.read_csv(file_path).astype(str)
    headers = data.columns.tolist()

    # Concatenate all fields in a row
    data['concat'] = data[headers].apply(lambda row: ''.join(row.values), axis=1)

    # Hash each concatenated string
    data['hash'] = data['concat'].astype(str).apply(
        lambda x: base64.b64encode(hashlib.sha256(x.encode('utf-8')).digest()).decode('utf-8')
    )

    unique_hash = data['hash'].unique()

    # Dummy column if missing
    if 'name' not in data.columns:
        data['name'] = data[headers[0]]
    if 'is_duplicate' not in data.columns:
        data['is_duplicate'] = data.duplicated().astype(int)
    data['is_duplicate'] = data['is_duplicate'].astype(int)
    # Split data
    train, test = train_test_split(data, test_size=0.05, stratify=data['is_duplicate'], random_state=42)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    # Deduplication model
    def deduplication_model(data, scoring_range=10, step=2):
        data['indices'] = list(range(len(data)))
        accuracy = []
        index = []
        final_step = 0

        for value in range(scoring_range):
            for i in unique_hash:
                sample = data[data.hash == i].reset_index(drop=True)
                for a in range(len(sample)):
                    comparison = sample[sample.indices != sample.indices[a]].reset_index(drop=True)
                    scores = [distance.levenshtein(sample.name[a], comparison.name[x]) for x in range(len(comparison))]
                    compare = [comparison.indices[x] for x in range(len(comparison)) if scores[x] <= value * step]
                    try:
                        if len(compare) > 0 and sample.indices[a] > compare[scores.index(min(scores))]:
                            index.append(sample.indices[a])
                    except ValueError:
                        pass
                    

            prediction = [1 if k in index else 0 for k in range(len(data))]
            data['prediction'] = prediction
            data = data.dropna(subset=['is_duplicate', 'prediction'])

            f1 = f1_score(data.is_duplicate, data.prediction)
            accuracy.append(f1)

            if len(accuracy) > 1 and accuracy[-1] <= accuracy[-2]:
                final_step += 1
            if final_step >= step:
                value = value - final_step
                break

        return prediction, value

    # Train
    performance, optimum = deduplication_model(train, scoring_range=10, step=2)

    # Prediction
    def deduplication_prediction(data, optimum_value):
        data['indices'] = list(range(len(data)))
        index = []
        for i in unique_hash:
            sample = data[data.hash == i].reset_index(drop=True)
            for a in range(len(sample)):
                comparison = sample[sample.indices != sample.indices[a]].reset_index(drop=True)
                scores = [distance.levenshtein(sample.name[a], comparison.name[x]) for x in range(len(comparison))]
                compare = [comparison.indices[x] for x in range(len(comparison))]
                try:
                    if len(compare) > 0 and sample.indices[a] > compare[scores.index(min(scores))]:
                        score = np.min(scores)
                        if score <= optimum_value:
                            index.append(sample.indices[a])
                except ValueError:
                    pass
        prediction = [1 if k in index else 0 for k in range(len(data))]
        data['prediction'] = prediction
        return prediction

    predictions = deduplication_prediction(test, optimum)
    acc = accuracy_score(test.is_duplicate, predictions)

    # Save output
    train['prediction'] = performance
    test['prediction'] = predictions
    dataset = pd.concat([train, test], axis=0)
    dataset = dataset[(dataset.prediction != 1)].reset_index(drop=True)
    dataset = dataset.drop(columns=['indices', 'concat', 'hash', 'prediction', 'name'], errors='ignore')
    # out = input("enter output file name with extension(.csv only):")
    # output_path = os.path.join('uploads', out)
    dataset.to_csv(output_path, index=False)

    end_h = time.time()
    time_taken = round(end_h - start_h, 2)
    result_text = f"✅ Deduplication Complete! Accuracy: {acc:.4f}"

    return result_text, output_path, time_taken


@app.route('/download/<path:filename>')
def download_file(filename):
    return send_file(filename, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
