import pandas as pd
from sentence_transformers import SentenceTransformer, util
from rouge_score import rouge_scorer

class RAGEvaluator:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.df = pd.read_csv(csv_file)
        
        if self.df.shape[1] < 3:
            raise ValueError("CSV file must contain at least 3 columns: Question, RAG Response, Ground Truth.")
        
        self.questions = self.df.iloc[:, 0].tolist()
        self.responses = self.df.iloc[:, 1].tolist()
        self.ground_truths = self.df.iloc[:, 2].tolist()
        
        self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        
        self.faithfulness_scores = []
        self.context_recall_scores = []
        self.context_precision_scores = []
    
    def cosine_similarity(self, text1, text2):
        emb1 = self.embedding_model.encode(text1, convert_to_tensor=True)
        emb2 = self.embedding_model.encode(text2, convert_to_tensor=True)
        return util.pytorch_cos_sim(emb1, emb2).item()
    
    def rouge_score(self, predicted, reference):
        scores = self.rouge_scorer.score(predicted, reference)
        return scores['rouge1'].fmeasure
    
    def evaluate(self):
        for question, response, ground_truth in zip(self.questions, self.responses, self.ground_truths):
            faithfulness = self.cosine_similarity(response, ground_truth)
            recall = self.cosine_similarity(ground_truth, response)
            precision = self.rouge_score(response, ground_truth)
            
            self.faithfulness_scores.append(faithfulness)
            self.context_recall_scores.append(recall)
            self.context_precision_scores.append(precision)
        
        avg_faithfulness = sum(self.faithfulness_scores) / len(self.faithfulness_scores)
        avg_recall = sum(self.context_recall_scores) / len(self.context_recall_scores)
        avg_precision = sum(self.context_precision_scores) / len(self.context_precision_scores)
        
        print("\n🔹 **RAG System Evaluation Results** 🔹")
        print(f"✅ Faithfulness Score (Cosine Similarity): {avg_faithfulness:.4f}")
        print(f"✅ Context Recall Score (Cosine Similarity): {avg_recall:.4f}")
        print(f"✅ Context Precision Score (ROUGE-1 F1): {avg_precision:.4f}")
        
        self.df["Faithfulness"] = self.faithfulness_scores
        self.df["Context Recall"] = self.context_recall_scores
        self.df["Context Precision"] = self.context_precision_scores
        self.df.to_csv("evaluated_results.csv", index=False)
        
        print("\n✅ Evaluation complete! Results saved to 'evaluated_results.csv'.")

if __name__ == "__main__":
    evaluator = RAGEvaluator("generated.csv")
    evaluator.evaluate()
