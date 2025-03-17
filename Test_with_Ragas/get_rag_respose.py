import pandas as pd
import requests
import json
import os

def fetch_answer(question, url="http://127.0.0.1:5000/ask"):
    try:
        payload = {"question": question}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json().get("answer", "No response")
        else:
            return f"Error: {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"

def process_csv(input_file, output_file):
    df = pd.read_csv(input_file)
    
    if "query" not in df.columns:
        print("Error: 'query' column not found in CSV file.")
        return
    
    for index, row in df.iterrows():
        query = row["query"]
        contest = row["contexts"]
        response = fetch_answer(query)
        new_data = pd.DataFrame([[query, response,contest]], columns=["query", "response","contexts"])
        
        # Append each new response to the file
        if os.path.exists(output_file):
            new_data.to_csv(output_file, mode='a', header=False, index=False)
        else:
            new_data.to_csv(output_file, index=False)
    
    print(f"Processed data appended to {output_file}")

# Example usage
input_csv = "output.csv"   # Replace with your CSV file
output_csv = "generated.csv" # Output file to append data
process_csv(input_csv, output_csv)