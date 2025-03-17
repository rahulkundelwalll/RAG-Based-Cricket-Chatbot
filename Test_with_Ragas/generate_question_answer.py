import csv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o")

# Read the articles from data.txt
input_file = "BBC_Sport_Output\data.txt"
output_file = "output.csv"

def process_article(article):
    """Generate two questions and answers from a given article."""
    prompt = f"Read the following article and generate two questions with their answers:\n\n{article}\n\nFormat: Question 1: <question> Answer 1: <answer> Question 2: <question> Answer 2: <answer>"
    
    messages = [
        SystemMessage("You are an expert in generating insightful questions and answers from articles."),
        HumanMessage(prompt)
    ]
    
    response = llm.invoke(messages)
    return response.content

# Process the file line by line and store results in CSV
with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", newline="", encoding="utf-8") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["query", "contexts"])  # Writing header
    
    article = ""
    for line in infile:
        if line.strip() == "================================================================================":
            if article:
                qa_pairs = process_article(article)
                for qa in qa_pairs.split("Question "):
                    if "Answer" in qa:
                        parts = qa.split("Answer ")
                        question = parts[0].strip().replace("1:", "").replace("2:", "")
                        answer = parts[1].strip()
                        writer.writerow([question, answer])
                article = ""  # Reset for next article
        else:
            article += line + " "

print("Processing complete. Data saved to output.csv")
