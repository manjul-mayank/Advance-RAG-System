from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('api/query/', views.query_rag, name='query_rag'),
    path('api/health/', views.health_check, name='health_check'),
    path('api/upload/', views.ingest_upload, name='ingest_upload'),
    path('api/clear-documents/', views.clear_all_documents, name='clear_documents'),  # NEW
    path('api/evaluate/', views.evaluate_response, name='evaluate_response'),
    path('api/benchmark/', views.benchmark_configurations, name='benchmark'),
    path('api/evaluate-auto/', views.evaluate_auto, name='evaluate_auto'),
    path('api/evaluation-metrics/', views.get_evaluation_metrics, name='evaluation_metrics'),
    path('api/evaluate-visual/', views.evaluate_with_visualization, name='evaluate_visual'),
    path('api/evaluation-dashboard/', views.get_evaluation_dashboard, name='evaluation_dashboard'),
    path('api/compare-evaluations/', views.compare_evaluations, name='compare_evaluations'),
]