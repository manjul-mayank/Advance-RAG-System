import os
from django.core.management.base import BaseCommand
from rag_api.utils import read_pdf, read_txt, split_text, save_metadata
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from django.conf import settings


class Command(BaseCommand):
    help = 'Ingest documents into vector store for RAG system'

    def add_arguments(self, parser):
        parser.add_argument(
            '--folder', 
            type=str, 
            required=True,
            help='Path to folder with documents (PDF, TXT, MD)'
        )
        parser.add_argument(
            '--out-dir', 
            type=str, 
            default=settings.VECTOR_DIR,
            help='Output directory for vector store (default: settings.VECTOR_DIR)'
        )
        parser.add_argument(
            '--model', 
            type=str, 
            default=settings.EMBEDDING_MODEL,
            help=f'Embedding model name (default: {settings.EMBEDDING_MODEL})'
        )
        parser.add_argument(
            '--chunk-size',
            type=int,
            default=500,
            help='Text chunk size in words (default: 500)'
        )
        parser.add_argument(
            '--chunk-overlap',
            type=int,
            default=50,
            help='Chunk overlap in words (default: 50)'
        )

    def handle(self, *args, **options):
        folder = options['folder']
        out_dir = options['out_dir']
        model_name = options['model']
        chunk_size = options['chunk_size']
        chunk_overlap = options['chunk_overlap']
        
        self.stdout.write(f"Starting ingestion from: {folder}")
        self.stdout.write(f"Output directory: {out_dir}")
        self.stdout.write(f"Using model: {model_name}")
        
        # Validate folder exists
        if not os.path.exists(folder):
            self.stderr.write(self.style.ERROR(f"Folder does not exist: {folder}"))
            return
        
        model = SentenceTransformer(model_name)
        texts, filenames, sources = [], [], []
        
        for root, _, files in os.walk(folder):
            for fn in files:
                if fn.startswith("."):
                    continue
                    
                path = os.path.join(root, fn)
                self.stdout.write(f"Processing: {fn}")
                
                try:
                    if fn.lower().endswith(".pdf"):
                        raw = read_pdf(path)
                    elif fn.lower().endswith(".txt") or fn.lower().endswith(".md"):
                        raw = read_txt(path)
                    else:
                        self.stdout.write(f"Skipping unsupported file: {fn}")
                        continue
                    
                    if not raw.strip():
                        self.stdout.write(f"Warning: Empty file: {fn}")
                        continue
                        
                    chunks = split_text(raw, chunk_size=chunk_size, overlap=chunk_overlap)
                    texts.extend(chunks)
                    filenames.extend([fn] * len(chunks))
                    sources.extend([path] * len(chunks))
                    
                    self.stdout.write(f"  → Extracted {len(chunks)} chunks")
                    
                except Exception as e:
                    self.stderr.write(self.style.ERROR(f"Error processing {fn}: {str(e)}"))
                    continue

        if not texts:
            self.stdout.write(self.style.WARNING('No text found in any documents.'))
            return

        self.stdout.write(f"Encoding {len(texts)} chunks...")
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        
        self.stdout.write("Creating FAISS index...")
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        
        os.makedirs(out_dir, exist_ok=True)
        faiss.write_index(index, os.path.join(out_dir, "faiss.index"))
        
        # Create metadata with source file paths
        metadata = []
        for i, (text, filename, source) in enumerate(zip(texts, filenames, sources)):
            metadata.append({
                "source": filename,
                "source_path": source,
                "text": text,
                "chunk": i
            })
        
        save_metadata(metadata, out_dir)
        
        self.stdout.write(
            self.style.SUCCESS(
                f'✅ Ingestion complete! Processed {len(texts)} chunks from {len(set(filenames))} files.'
            )
        )
        self.stdout.write(f'📁 Vector store saved to: {out_dir}')