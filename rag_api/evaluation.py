import numpy as np
import json
import logging
from typing import List, Dict, Any
from sentence_transformers import util
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime

logger = logging.getLogger(__name__)

class RAGEvaluator:
    def __init__(self):
        # Try to import optional dependencies
        self.rouge_available = False
        self.evaluation_history = []
        try:
            from rouge_score import rouge_scorer
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            self.rouge_available = True
        except ImportError:
            logger.warning("rouge-score not available. Install with: pip install rouge-score")
    
    def calculate_retrieval_metrics(self, retrieved_docs: List[str], relevant_docs: List[str]) -> Dict[str, float]:
        """Calculate retrieval precision, recall, and F1"""
        if not retrieved_docs or not relevant_docs:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            
        # Simple exact match for relevance
        retrieved_set = set([doc.lower().strip() for doc in retrieved_docs])
        relevant_set = set([doc.lower().strip() for doc in relevant_docs])
        
        true_positives = len(retrieved_set.intersection(relevant_set))
        false_positives = len(retrieved_set - relevant_set)
        false_negatives = len(relevant_set - retrieved_set)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives
        }
    
    def calculate_semantic_similarity(self, text1: str, text2: str, embed_model) -> float:
        """Calculate semantic similarity using embedding model"""
        try:
            emb1 = embed_model.encode([text1], convert_to_tensor=True)
            emb2 = embed_model.encode([text2], convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(emb1, emb2).item()
            return round(similarity, 4)
        except Exception as e:
            logger.warning(f"Semantic similarity calculation failed: {e}")
            return 0.0
    
    def calculate_rouge_scores(self, generated: str, reference: str) -> Dict[str, float]:
        """Calculate ROUGE scores for text generation quality"""
        if not self.rouge_available:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "note": "rouge-score not installed"}
            
        try:
            scores = self.rouge_scorer.score(reference, generated)
            return {
                "rouge1": round(scores['rouge1'].fmeasure, 4),
                "rouge2": round(scores['rouge2'].fmeasure, 4),
                "rougeL": round(scores['rougeL'].fmeasure, 4)
            }
        except Exception as e:
            logger.warning(f"ROUGE calculation failed: {e}")
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    
    def calculate_answer_relevance(self, query: str, answer: str, embed_model) -> float:
        """Calculate how relevant the answer is to the query"""
        return self.calculate_semantic_similarity(query, answer, embed_model)
    
    def calculate_context_utilization(self, context: str, answer: str, embed_model) -> float:
        """Calculate how much the answer utilizes the provided context"""
        return self.calculate_semantic_similarity(context, answer, embed_model)
    
    def calculate_faithfulness(self, context: str, answer: str, embed_model, threshold: float = 0.7) -> float:
        """Calculate if the answer is faithful to the context (avoiding hallucination)"""
        # Split context and answer into sentences for more granular comparison
        context_sents = [s.strip() for s in context.split('.') if s.strip()]
        answer_sents = [s.strip() for s in answer.split('.') if s.strip()]
        
        if not answer_sents:
            return 0.0
            
        faithful_sents = 0
        for ans_sent in answer_sents:
            max_similarity = 0
            for ctx_sent in context_sents:
                similarity = self.calculate_semantic_similarity(ans_sent, ctx_sent, embed_model)
                max_similarity = max(max_similarity, similarity)
            
            if max_similarity >= threshold:
                faithful_sents += 1
                
        return round(faithful_sents / len(answer_sents), 4)
    
    def calculate_comprehensiveness(self, query: str, answer: str, context: str, embed_model) -> float:
        """Calculate if answer covers important aspects from context relevant to query"""
        # Simplified version using semantic similarity
        query_context_sim = self.calculate_semantic_similarity(query, context, embed_model)
        query_answer_sim = self.calculate_semantic_similarity(query, answer, embed_model)
        
        # If answer is as relevant to query as context is, it's comprehensive
        return round(min(query_answer_sim / (query_context_sim + 1e-8), 1.0), 4)
    
    def evaluate_rag_response(self, query: str, context: str, answer: str, 
                            retrieved_docs: List[str], relevant_docs: List[str],
                            embed_model) -> Dict[str, Any]:
        """Comprehensive RAG evaluation"""
        
        # Retrieval metrics
        retrieval_metrics = self.calculate_retrieval_metrics(retrieved_docs, relevant_docs)
        
        # Generation metrics
        rouge_scores = self.calculate_rouge_scores(answer, self._create_ideal_answer(context, query))
        
        # Quality metrics
        answer_relevance = self.calculate_answer_relevance(query, answer, embed_model)
        context_utilization = self.calculate_context_utilization(context, answer, embed_model)
        faithfulness = self.calculate_faithfulness(context, answer, embed_model)
        comprehensiveness = self.calculate_comprehensiveness(query, answer, context, embed_model)
        
        # Overall scores
        quality_score = np.mean([answer_relevance, context_utilization, faithfulness, comprehensiveness])
        retrieval_score = retrieval_metrics['f1']
        generation_score = np.mean([v for v in rouge_scores.values() if isinstance(v, (int, float))])
        
        overall_score = np.mean([quality_score, retrieval_score, generation_score])
        
        return {
            "overall_score": round(overall_score, 4),
            "breakdown": {
                "quality_score": round(quality_score, 4),
                "retrieval_score": round(retrieval_score, 4),
                "generation_score": round(generation_score, 4),
            },
            "retrieval_metrics": retrieval_metrics,
            "generation_metrics": rouge_scores,
            "quality_metrics": {
                "answer_relevance": answer_relevance,
                "context_utilization": context_utilization,
                "faithfulness": faithfulness,
                "comprehensiveness": comprehensiveness,
            },
            "dependencies_available": {
                "rouge": self.rouge_available,
            },
            "summary": self._get_quality_summary(overall_score)
        }
    
    def _create_ideal_answer(self, context: str, query: str) -> str:
        """Create a simple ideal answer for ROUGE comparison"""
        return f"Based on the context provided, the answer to '{query}' can be found in the information given."
    
    def _get_quality_summary(self, score: float) -> str:
        if score >= 0.9:
            return "Excellent - Highly accurate and comprehensive"
        elif score >= 0.7:
            return "Good - Reliable with minor issues"
        elif score >= 0.5:
            return "Fair - Some accuracy or relevance issues"
        else:
            return "Poor - Significant quality concerns"
    
    def benchmark_configurations(self, test_queries: List[Dict], configurations: List[Dict]) -> Dict[str, Any]:
        """Benchmark different RAG configurations"""
        results = {}
        
        for config in configurations:
            config_name = f"{config.get('model', 'default')}_{config.get('chunking', 'default')}_{config.get('prompting', 'basic')}"
            scores = []
            
            for query_data in test_queries:
                # In a real implementation, you would run the actual RAG system with this config
                # For now, we'll use a mock evaluation
                mock_eval = {
                    "overall_score": np.random.uniform(0.5, 0.9)  # Mock score
                }
                scores.append(mock_eval["overall_score"])
            
            results[config_name] = {
                "average_score": round(np.mean(scores), 4),
                "std_score": round(np.std(scores), 4),
                "min_score": round(np.min(scores), 4),
                "max_score": round(np.max(scores), 4),
                "config": config
            }
        
        # Rank configurations
        ranked_results = dict(sorted(results.items(), key=lambda x: x[1]["average_score"], reverse=True))
        
        return {
            "benchmark_results": ranked_results,
            "best_configuration": list(ranked_results.keys())[0],
            "best_score": list(ranked_results.values())[0]["average_score"],
            "note": "Benchmarking uses mock data. Integrate with actual RAG system for real results."
        }
        
    def generate_evaluation_chart(self, evaluation_data: Dict[str, Any]) -> str:
        """Generate a radar chart for evaluation metrics"""
        try:
            # Prepare data for radar chart
            categories = ['Retrieval', 'Answer Relevance', 'Context Usage', 'Faithfulness', 'Comprehensiveness']
            scores = [
                evaluation_data['breakdown']['retrieval_score'],
                evaluation_data['quality_metrics']['answer_relevance'],
                evaluation_data['quality_metrics']['context_utilization'], 
                evaluation_data['quality_metrics']['faithfulness'],
                evaluation_data['quality_metrics']['comprehensiveness']
            ]
            
            # Create radar chart
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
            
            # Complete the circle
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            scores += scores[:1]
            angles += angles[:1]
            categories_radar = categories + [categories[0]]
            
            # Plot
            ax.plot(angles, scores, 'o-', linewidth=2, label='RAG Performance')
            ax.fill(angles, scores, alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_ylim(0, 1)
            ax.set_title('RAG Evaluation Radar Chart', size=14, fontweight='bold')
            ax.grid(True)
            
            # Save to base64 string
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            
            plt.close(fig)
            return base64.b64encode(image_png).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Chart generation failed: {e}")
            return ""
    
    def generate_metric_bars(self, evaluation_data: Dict[str, Any]) -> str:
        """Generate bar chart for individual metrics"""
        try:
            metrics = {
                'Overall Score': evaluation_data['overall_score'],
                'Retrieval F1': evaluation_data['retrieval_metrics']['f1'],
                'Answer Relevance': evaluation_data['quality_metrics']['answer_relevance'],
                'Context Utilization': evaluation_data['quality_metrics']['context_utilization'],
                'Faithfulness': evaluation_data['quality_metrics']['faithfulness'],
                'ROUGE-L': evaluation_data['generation_metrics'].get('rougeL', 0)
            }
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(metrics.keys(), metrics.values(), color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3E92CC', '#6A8EAE'])
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom')
            
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1)
            ax.set_title('RAG Evaluation Metrics', fontweight='bold')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            
            plt.close(fig)
            return base64.b64encode(image_png).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Bar chart generation failed: {e}")
            return ""
    
    def generate_comparison_chart(self, evaluations: List[Dict]) -> str:
        """Generate comparison chart for multiple evaluations"""
        try:
            labels = [f"Eval {i+1}" for i in range(len(evaluations))]
            overall_scores = [e['overall_score'] for e in evaluations]
            retrieval_scores = [e['breakdown']['retrieval_score'] for e in evaluations]
            quality_scores = [e['breakdown']['quality_score'] for e in evaluations]
            
            x = np.arange(len(labels))
            width = 0.25
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(x - width, overall_scores, width, label='Overall', color='#2E86AB')
            ax.bar(x, retrieval_scores, width, label='Retrieval', color='#A23B72') 
            ax.bar(x + width, quality_scores, width, label='Quality', color='#F18F01')
            
            ax.set_xlabel('Evaluations')
            ax.set_ylabel('Scores')
            ax.set_title('RAG Evaluation Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.legend()
            ax.set_ylim(0, 1)
            
            plt.tight_layout()
            
            # Save to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            
            plt.close(fig)
            return base64.b64encode(image_png).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Comparison chart generation failed: {e}")
            return ""
    
    def store_evaluation(self, evaluation_data: Dict[str, Any]):
        """Store evaluation in history for trend analysis"""
        self.evaluation_history.append({
            'timestamp': datetime.now().isoformat(),
            'data': evaluation_data
        })
        # Keep only last 50 evaluations
        if len(self.evaluation_history) > 50:
            self.evaluation_history = self.evaluation_history[-50:]
    
    def generate_trend_chart(self) -> str:
        """Generate trend analysis of evaluation history"""
        if len(self.evaluation_history) < 2:
            return ""
            
        try:
            timestamps = [e['timestamp'] for e in self.evaluation_history]
            overall_scores = [e['data']['overall_score'] for e in self.evaluation_history]
            
            # Convert timestamps to datetime objects
            dates = [datetime.fromisoformat(ts) for ts in timestamps]
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(dates, overall_scores, 'o-', linewidth=2, markersize=4, color='#2E86AB')
            ax.set_xlabel('Time')
            ax.set_ylabel('Overall Score')
            ax.set_title('RAG Performance Trend Over Time')
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            
            plt.close(fig)
            return base64.b64encode(image_png).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Trend chart generation failed: {e}")
            return ""