# RAG-Based Cricket Chatbot

## Introduction

This project is a **Retrieval-Augmented Generation (RAG)**-based chatbot designed to provide accurate cricket-related responses by retrieving relevant information before generating answers using **Ollama LLaMA 3** and **all-MiniLM-L6-v2** for embeddings.

---

## Table of Contents

- [What is RAG?](#what-is-rag)
- [What is Ollama?](#what-is-ollama)
- [What is LLaMA 3?](#what-is-llama-3)
- [How the Chatbot Works](#how-the-chatbot-works)
- [Threshold Tuning](#threshold-tuning)
- [Project Structure](#project-structure)
- [Data Source](#data-source)
- [Installation and Setup](#installation-and-setup)
- [Running the Chatbot](#running-the-chatbot)
- [Customization](#customization)
- [Evaluation](#evaluation)
- [Future Enhancements](#future-enhancements)
- [Conclusion](#conclusion)

---

## What is RAG?

Retrieval-Augmented Generation (RAG) is an advanced AI technique that enhances language models by integrating a retrieval mechanism before generating responses. It consists of:

1. **Retrieval Component**: Searches a document store or knowledge base to fetch relevant information.
2. **Generation Component**: Uses a language model (e.g., Ollama LLaMA 3) to generate responses based on retrieved data.

This approach improves response accuracy, ensures factual correctness, and enhances contextual understanding.

---

## What is Ollama?

**Ollama** is an AI framework that allows running large language models (LLMs) efficiently on local machines. It provides optimized inference for models like LLaMA 3, enabling users to integrate AI-driven conversations in their applications.

---

## What is LLaMA 3?

LLaMA 3 (Large Language Model Meta AI) is an advanced open-weight transformer-based model designed by Meta AI. It excels in natural language processing tasks, making it suitable for applications such as chatbots and knowledge retrieval.

---

## How the Chatbot Works

The chatbot follows these key steps:

### 1. **User Query Processing**

- The user enters a query related to cricket.
- The query undergoes preprocessing (text normalization, tokenization, etc.).

### 2. **Retrieval Phase**

- The query is converted into an embedding using **all-MiniLM-L6-v2**.
- A similarity search is performed in the document store (ChromaDB) to fetch relevant cricket-related information.

### 3. **Augmentation Phase**

- The retrieved data is formatted and combined with the original query.
- The augmented query is sent to **Ollama LLaMA 3** for processing.

### 4. **Response Generation**

- LLaMA 3 generates a response using both retrieved data and its internal knowledge.
- The response undergoes post-processing before being sent to the user.

### 5. **Threshold Tuning**

- A similarity threshold is used to determine if retrieved information is relevant.
- If the similarity score is below the threshold, fallback mechanisms are triggered.

---

## Threshold Tuning

The chatbot uses a **retrieval threshold** to determine when to use retrieved data.

- **Higher threshold** (e.g., 0.85 - 0.9): More accurate but may reject some relevant queries.
- **Lower threshold** (e.g., 0.6 - 0.7): More inclusive but might allow less relevant responses.

You can configure the threshold in `config.py`:

```python
RETRIEVAL_THRESHOLD = 0.3  # Adjust based on performance needs
```

---

## Project Structure

```
BBC_scrap/
│-- webScrape.py          # scrap 800 pages from website
BBC_scrap/
│-- data.text            # scrap data into txt
db/                      # chunks
RAG_Pipeline/
│-- server.py            # Main server to handle user interactions
│-- Split_data_in_chunk  # Splits data into chunks
│-- ChromaDBRetriber.py  # Stores data into ChromaDB
│-- Rag.py               # Retrieves data and feeds it to the LLM
RAG_Pipeline/
│-- server.py            # Main server to handle user interactions
│-- Split_data_in_chunk  # Splits data into chunks
│-- ChromaDBRetriber.py  # Stores data into ChromaDB
│-- Rag.py               # Retrieves data and feeds it to the LLM
Test_with_Ragas/
│-- test.py              #evaluation
```

---

## Data Source

The chatbot retrieves cricket-related articles from **Indian Express**.

- **Scraped Data Source:** [Indian Express Cricket Section](https://indianexpress.com/section/sports/cricket)
- **Number of Articles:** ~800 pages
- **Storage:** The articles are stored in **ChromaDB** for efficient retrieval.

---

## Installation and Setup

### 1. Clone the repository:

```bash
git clone https://github.com/rahulkundelwalll/criket_news_chatbot
cd criket_news_chatbot
```

### 2. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Chatbot

To start the chatbot, run the following command:

```bash
python RAG_Pipeline/server.py
```

The chatbot will be available at `http://localhost:5000` (or the configured port).

---

## Customization

### **Modify Retrieval Parameters**

Adjust retrieval settings in `ChromaDBRetriber.py`:

```python
def retrieve_documents(query, top_k=5):
    # Fetch top_k relevant documents
```

---

## Evaluation

The chatbot’s responses can be evaluated using a **faithfulness, recall, and precision** scoring system:

```bash
python Test_with_Ragas/server.py
```

### **Evaluation Metrics:**

| Metric                | Definition                                                                      |
| --------------------- | ------------------------------------------------------------------------------- |
| **Faithfulness**      | Measures cosine similarity between the chatbot’s response and the ground truth. |
| **Context Recall**    | Measures similarity between ground truth and response.                          |
| **Context Precision** | Uses ROUGE-1 F1 to measure overlap.                                             |

Example output:

```
🔹 **RAG System Evaluation Results** 🔹
✅ Faithfulness Score (Cosine Similarity): 0.7195
✅ Context Recall Score (Cosine Similarity): 0.7195
✅ Context Precision Score (ROUGE-1 F1): 0.3396
```

Results are saved in `evaluated_results.csv`.

---

## Future Enhancements

1. **Improve Retrieval Accuracy:** Use advanced vector search techniques.
2. **Model Fine-tuning:** Optimize LLaMA 3 for cricket-specific conversations.
3. **UI Development:** Add a web interface for better user experience.
4. **Multilingual Support:** Expand chatbot capabilities to support multiple languages.

---

## Conclusion

This RAG-based cricket chatbot effectively combines retrieval and generative AI techniques to provide **accurate** and **contextually relevant** responses. By fine-tuning the retrieval threshold, expanding the knowledge base, and refining prompt engineering, the chatbot can be further optimized for **better performance**.
