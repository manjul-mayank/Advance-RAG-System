```bash




💡 PRO TIPS
Always activate virtual environment first

Use --noreload for production-like testing

Set performance env variables for better stability

Clear cache regularly during development

Monitor memory usage with large models

Save this as SERVER_COMMANDS.md in your project root for quick reference! 📋



## 🔧 Virtual Environment & Setup ##

# Activate virtual environment
source rag_3_13/bin/activate

# Deactivate virtual environment
deactivate

# Install requirements
pip install -r requirements.txt

# Freeze current dependencies
pip freeze > requirements.txt

## 🔄Server Operations ##

# Normal server start
python manage.py runserver

# Full reload server (with optimizations)
source rag_3_13/bin/activate
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
python manage.py runserver --noreload

# Server on specific port
python manage.py runserver 8080

# Server on all interfaces
python manage.py runserver 0.0.0.0:8000

## ⚡ Process Management ##

# Kill server on port 8000
lsof -ti:8000 | xargs kill -9

# Kill all Python processes
pkill -f python

# Find what's running on port 8000
lsof -i :8000

# Check if server is running
ps aux | grep runserver

## 🧹 Cache & Cleanup ##

# Clear Python cache
find . -name "__pycache__" -type d -exec rm -rf {} +

# Clear all .pyc files
find . -name "*.pyc" -delete

# Clear migrations (careful!)
find . -path "*/migrations/*.py" -not -name "__init__.py" -delete

# Clear database (if needed)
rm db.sqlite3

## 📊 Data Management ##

# Ingest files to vector store
python manage.py ingest --folder data/ --out-dir vector_store/

# Ingest single file
python manage.py ingest --file document.pdf --out-dir vector_store/

# Rebuild FAISS index
python manage.py rebuild_index

# Check vector store status
python manage.py vector_status

## 🔍 System Diagnostics ##

# Check system health via API
curl http://localhost:8000/api/health/

# Test evaluation endpoint
curl -X POST http://localhost:8000/api/evaluate/ \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "context": "test", "answer": "test"}'

# Check database
python manage.py dbshell

# Show migrations
python manage.py showmigrations

## ⚙️ Performance Optimizations ##

# Enable MPS fallback for Apple Silicon
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Disable MPS (if having issues)
export PYTORCH_MPS_DISABLE=1

# CPU-only mode
export CUDA_VISIBLE_DEVICES=""

# Memory optimization
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

## 🐛 Debugging & Logs ##

# Run with debug mode
python manage.py runserver --verbosity 2

# Check logs in real-time
tail -f logs/debug.log

# Test specific component
python manage.py test rag_api.tests

# Database migrations
python manage.py makemigrations
python manage.py migrate

## 📈 Monitoring ##

# Check memory usage
htop

# Monitor GPU usage (if available)
nvidia-smi

# Check disk space
df -h

# Monitor network connections
netstat -tulpn | grep :8000

## 🛠️ Development Tools ##

# Django shell
python manage.py shell

# Create superuser
python manage.py createsuperuser

# Collect static files
python manage.py collectstatic

# Check for broken links
python manage.py check

## 🎯 QUICK REFERENCE CARD ##

# 🚀 START SERVER
source rag_3_13/bin/activate
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false python manage.py runserver --noreload

# 🛑 STOP SERVER  
lsof -ti:8000 | xargs kill -9

# 📥 INGEST DATA
python manage.py ingest --folder data/ --out-dir vector_store/

# 🧹 CLEANUP
find . -name "__pycache__" -type d -exec rm -rf {} +

# 🔍 CHECK HEALTH
curl http://localhost:8000/api/health/
