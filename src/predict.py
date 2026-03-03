import pandas as pd
import joblib
import os

MODEL_FILE = "model/model.pkl"
PIPELINE_FILE = "model/pipeline.pkl"

INPUT_FILE = "data/input_sample.csv"
OUTPUT_FILE = "data/output.csv"

# check if model exists
if not os.path.exists(MODEL_FILE):
    print("Model file not found. Please run train.py first.")
    exit()

# load model and pipeline
model = joblib.load(MODEL_FILE)
pipeline = joblib.load(PIPELINE_FILE)

# load input data
input_data = pd.read_csv(INPUT_FILE)

# if label exists, separate it (optional case)
if "median_house_value" in input_data.columns:
    input_features = input_data.drop("median_house_value", axis=1)
else:
    input_features = input_data.copy()

# transform data
transformed_input = pipeline.transform(input_features)

# make predictions
predictions = model.predict(transformed_input)

# attach predictions
input_data["predicted_median_house_value"] = predictions

# save output
input_data.to_csv(OUTPUT_FILE, index=False)

print("Inference is complete. Output saved as output.csv")