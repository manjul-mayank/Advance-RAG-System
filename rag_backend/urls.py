from django.contrib import admin
from rag_api import views
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path("admin/", admin.site.urls),
    path('', include('rag_api.urls')), 
    path("", views.index, name="index"),
    path("api/query/", views.query_rag, name="query_rag"),
    path("api/health/", views.health_check, name="health_check"),
]