import os
import django
import sys

# Add your project to Python path
sys.path.append('/Users/manjulmayank/RAG-for-local-engine')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'rag_backend.settings')
django.setup()

from rag_api.evaluation import RAGEvaluator
from sentence_transformers import SentenceTransformer
import json

def test_evaluation_locally():
    print("🧪 Testing RAG Evaluation System Locally...")
    
    # Initialize components
    evaluator = RAGEvaluator()
    embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Test data
    test_data = {
        "query": "What is machine learning?",
        "context": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves.",
        "answer": "Machine learning is a branch of artificial intelligence that allows computers to learn from data and improve their performance on tasks without explicit programming.",
        "retrieved_docs": ["ai_textbook.pdf", "ml_research.pdf"],
        "relevant_docs": ["ai_textbook.pdf", "ml_research.pdf", "ai_basics.pdf"]
    }
    
    print(f"📊 Testing with query: {test_data['query']}")
    
    # Run evaluation
    evaluation = evaluator.evaluate_rag_response(
        query=test_data["query"],
        context=test_data["context"],
        answer=test_data["answer"],
        retrieved_docs=test_data["retrieved_docs"],
        relevant_docs=test_data["relevant_docs"],
        embed_model=embed_model
    )
    
    # Print numerical results
    print("\n" + "="*50)
    print("📈 NUMERICAL EVALUATION RESULTS:")
    print("="*50)
    
    print(f"🎯 Overall Score: {evaluation['overall_score']:.3f}")
    print(f"📋 Summary: {evaluation['summary']}")
    
    print("\n�� Breakdown Scores:")
    print(f"  • Quality Score: {evaluation['breakdown']['quality_score']:.3f}")
    print(f"  • Retrieval Score: {evaluation['breakdown']['retrieval_score']:.3f}")
    print(f"  • Generation Score: {evaluation['breakdown']['generation_score']:.3f}")
    
    print("\n🎯 Retrieval Metrics:")
    retrieval = evaluation['retrieval_metrics']
    print(f"  • Precision: {retrieval['precision']:.3f}")
    print(f"  • Recall: {retrieval['recall']:.3f}")
    print(f"  • F1 Score: {retrieval['f1']:.3f}")
    print(f"  • True Positives: {retrieval['true_positives']}")
    print(f"  • False Positives: {retrieval['false_positives']}")
    
    print("\n💡 Quality Metrics:")
    quality = evaluation['quality_metrics']
    print(f"  • Answer Relevance: {quality['answer_relevance']:.3f}")
    print(f"  • Context Utilization: {quality['context_utilization']:.3f}")
    print(f"  • Faithfulness: {quality['faithfulness']:.3f}")
    print(f"  • Comprehensiveness: {quality['comprehensiveness']:.3f}")
    
    print("\n📝 Generation Metrics:")
    generation = evaluation['generation_metrics']
    for metric, score in generation.items():
        if isinstance(score, (int, float)):
            print(f"  • {metric.upper()}: {score:.3f}")

def base64_to_image(base64_string):
    """Convert base64 string to image bytes"""
    import base64
    return base64.b64decode(base64_string)

if __name__ == "__main__":
    test_evaluation_locally()
