import os
from pathlib import Path

# ===============================================
# 📂 BASE PATHS & SECURITY
# ===============================================
BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = "django-insecure-local-rag-secret-key"
DEBUG = False
ALLOWED_HOSTS = ["*"]  # allow all for local use

# ===============================================
# ⚙️ INSTALLED APPS
# ===============================================
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "rest_framework",
    "corsheaders",   # ✅ allows CORS for API requests
    "rag_api",       # your local RAG app
]

# ===============================================
# 🔐 MIDDLEWARE
# ===============================================
MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",  # must come before CommonMiddleware
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

# ===============================================
# 🌐 CORS / CSRF SETTINGS
# ===============================================
# Allow local frontend access (avoids ❌ "Failed to fetch")
CORS_ALLOW_ALL_ORIGINS = True
CORS_ALLOW_HEADERS = [
    "accept",
    "accept-encoding",
    "authorization",
    "content-type",
    "dnt",
    "origin",
    "user-agent",
    "x-csrftoken",
    "x-requested-with",
]

CORS_ALLOW_METHODS = ["GET", "POST", "OPTIONS"]
CSRF_TRUSTED_ORIGINS = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

# ===============================================
# 🧭 URL & TEMPLATE CONFIGURATION
# ===============================================
ROOT_URLCONF = "rag_backend.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [os.path.join(BASE_DIR, "templates")],  # ✅ frontend page
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "rag_backend.wsgi.application"

# ===============================================
# 🗄️ DATABASE
# ===============================================
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

# ===============================================
# 🧱 STATIC FILES
# ===============================================
STATIC_URL = "/static/"
STATICFILES_DIRS = [
    os.path.join(BASE_DIR, "static"),
]

# ===============================================
# 🧠 RAG CONFIGURATION
# ===============================================
# RAG Configuration
VECTOR_DIR = os.path.join(BASE_DIR, 'vector_store')
DATA_DIR = os.path.join(BASE_DIR, 'data')
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
GEN_MODEL = 'microsoft/DialoGPT-medium'
TOP_K = 5

# Create directories if they don't exist
os.makedirs(VECTOR_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ===============================================
# 🧩 REST FRAMEWORK (Optional but Recommended)
# ===============================================
REST_FRAMEWORK = {
    "DEFAULT_RENDERER_CLASSES": [
        "rest_framework.renderers.JSONRenderer",
    ],
    "DEFAULT_PARSER_CLASSES": [
        "rest_framework.parsers.JSONParser",
        "rest_framework.parsers.FormParser",
        "rest_framework.parsers.MultiPartParser",
    ],
}

# ===============================================
# ✅ LOGGING (Helpful for debugging)
# ===============================================
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "console": {"class": "logging.StreamHandler"},
    },
    "root": {
        "handlers": ["console"],
        "level": "INFO",
    },
}

print(f"✅ Settings loaded successfully. BASE_DIR = {BASE_DIR}")

# ===============================================
# 🚀 PRODUCTION SETTINGS FOR CLOUD RUN
# ===============================================
import os

# Use /tmp for SQLite in Cloud Run (has write access)
if os.getenv('GAE_APPLICATION', None):
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.sqlite3',
            'NAME': '/tmp/db.sqlite3',
        }
    }
    
    # Add WhiteNoise for static files
    MIDDLEWARE.insert(1, 'whitenoise.middleware.WhiteNoiseMiddleware')
    
    # Static files configuration
    STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')
    
    # For RAG - use /tmp directory
    VECTOR_DIR = '/tmp/vector_store'
    DATA_DIR = '/tmp/data'
    
    # Create directories
    os.makedirs(VECTOR_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
