#!/usr/bin/env python3
"""
PDF Ingestion Script for Shankh.ai RAG Service

This script ingests PDF documents, chunks text, generates multilingual embeddings,
and builds a FAISS vector index for semantic search.

Usage:
    python ingest.py --data-dir ../../data --output-dir ./index
    python ingest.py --data-dir ../../data --chunk-size 700 --overlap 100

Author: Shankh.ai Team
"""

import os
import json
import argparse
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# PDF processing libraries (multiple for robustness)
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    print("Warning: pdfplumber not available, falling back to pypdf")

try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    try:
        from PyPDF2 import PdfReader
        PYPDF_AVAILABLE = True
    except ImportError:
        PYPDF_AVAILABLE = False
        print("Error: No PDF library available. Install pypdf or PyPDF2.")

# Load environment variables
load_dotenv()


class DocumentChunk:
    """Represents a text chunk with metadata"""
    def __init__(self, text: str, filename: str, page_num: int, 
                 chunk_id: int, char_start: int, char_end: int):
        self.text = text.strip()
        self.filename = filename
        self.page_num = page_num
        self.chunk_id = chunk_id
        self.char_start = char_start
        self.char_end = char_end
        self.excerpt = self.text[:100] + "..." if len(self.text) > 100 else self.text
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "text": self.text,
            "filename": self.filename,
            "page_num": self.page_num,
            "chunk_id": self.chunk_id,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "excerpt": self.excerpt
        }


class PDFIngestionPipeline:
    """Complete pipeline for PDF ingestion and vector index creation"""
    
    def __init__(self, 
                 embedding_model: str = None,
                 chunk_size: int = 700,
                 chunk_overlap: int = 100):
        """
        Initialize the ingestion pipeline
        
        Args:
            embedding_model: Name of sentence-transformer model (default from env)
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap between consecutive chunks
        """
        self.embedding_model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL", 
            "paraphrase-multilingual-mpnet-base-v2"
        )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        print(f"Initializing embedding model: {self.embedding_model_name}")
        print(f"This may take a few minutes on first run (downloading model)...")
        
        # Load sentence transformer model without authentication
        # Use token=False to avoid authentication issues
        self.model = SentenceTransformer(
            self.embedding_model_name,
            token=False  # Don't use HuggingFace token for public models
        )
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        print(f"✓ Model loaded (embedding dimension: {self.embedding_dim})")
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Tuple[int, str]]:
        """
        Extract text from PDF file, returns list of (page_num, text) tuples
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of (page_number, page_text) tuples
        """
        pages = []
        filename = Path(pdf_path).name
        
        try:
            # Try pdfplumber first (better text extraction)
            if PDFPLUMBER_AVAILABLE:
                print(f"  Using pdfplumber for {filename}")
                with pdfplumber.open(pdf_path) as pdf:
                    for page_num, page in enumerate(pdf.pages, start=1):
                        text = page.extract_text()
                        if text:
                            pages.append((page_num, text))
            
            # Fallback to pypdf/PyPDF2
            elif PYPDF_AVAILABLE:
                print(f"  Using pypdf for {filename}")
                reader = PdfReader(pdf_path)
                for page_num, page in enumerate(reader.pages, start=1):
                    text = page.extract_text()
                    if text:
                        pages.append((page_num, text))
            else:
                raise RuntimeError("No PDF library available")
            
            print(f"  ✓ Extracted {len(pages)} pages from {filename}")
            return pages
            
        except Exception as e:
            print(f"  ✗ Error extracting text from {filename}: {e}")
            return []
    
    def chunk_text(self, text: str, filename: str, page_num: int, 
                   chunk_offset: int = 0) -> List[DocumentChunk]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Text to chunk
            filename: Source filename
            page_num: Page number
            chunk_offset: Starting chunk ID offset
            
        Returns:
            List of DocumentChunk objects
        """
        chunks = []
        text_len = len(text)
        start = 0
        chunk_id = chunk_offset
        
        while start < text_len:
            # Calculate end position
            end = start + self.chunk_size
            
            # If not at the end, try to break at sentence boundary
            if end < text_len:
                # Look for sentence endings within last 100 chars
                search_start = max(start, end - 100)
                sentence_ends = [
                    text.rfind('. ', search_start, end),
                    text.rfind('। ', search_start, end),  # Hindi sentence end
                    text.rfind('? ', search_start, end),
                    text.rfind('! ', search_start, end),
                    text.rfind('\n\n', search_start, end),
                ]
                sentence_end = max(sentence_ends)
                
                if sentence_end > start:
                    end = sentence_end + 1
            
            chunk_text = text[start:end].strip()
            
            # Only create chunk if it has meaningful content
            if len(chunk_text) > 50:  # Minimum chunk size
                chunk = DocumentChunk(
                    text=chunk_text,
                    filename=filename,
                    page_num=page_num,
                    chunk_id=chunk_id,
                    char_start=start,
                    char_end=end
                )
                chunks.append(chunk)
                chunk_id += 1
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            
            # Prevent infinite loop
            if start >= text_len - self.chunk_overlap:
                break
        
        return chunks
    
    def process_pdfs(self, data_dir: str) -> List[DocumentChunk]:
        """
        Process all PDFs in directory and create chunks
        
        Args:
            data_dir: Directory containing PDF files
            
        Returns:
            List of all DocumentChunk objects
        """
        data_path = Path(data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        pdf_files = list(data_path.glob("*.pdf"))
        if not pdf_files:
            raise ValueError(f"No PDF files found in {data_dir}")
        
        print(f"\nFound {len(pdf_files)} PDF files:")
        for pdf in pdf_files:
            print(f"  - {pdf.name}")
        
        all_chunks = []
        chunk_id_offset = 0
        
        for pdf_path in pdf_files:
            print(f"\nProcessing: {pdf_path.name}")
            pages = self.extract_text_from_pdf(str(pdf_path))
            
            for page_num, page_text in pages:
                chunks = self.chunk_text(
                    page_text, 
                    pdf_path.name, 
                    page_num,
                    chunk_offset=chunk_id_offset
                )
                all_chunks.extend(chunks)
                chunk_id_offset += len(chunks)
                print(f"    Page {page_num}: {len(chunks)} chunks")
        
        print(f"\n✓ Total chunks created: {len(all_chunks)}")
        return all_chunks
    
    def create_embeddings(self, chunks: List[DocumentChunk]) -> np.ndarray:
        """
        Generate embeddings for all chunks
        
        Args:
            chunks: List of DocumentChunk objects
            
        Returns:
            Numpy array of embeddings (n_chunks x embedding_dim)
        """
        print(f"\nGenerating embeddings for {len(chunks)} chunks...")
        print("This may take several minutes depending on corpus size...")
        
        texts = [chunk.text for chunk in chunks]
        
        # Generate embeddings in batches for efficiency
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        print(f"✓ Generated embeddings shape: {embeddings.shape}")
        return embeddings
    
    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.IndexFlatIP:
        """
        Build FAISS index for similarity search
        
        Args:
            embeddings: Numpy array of embeddings
            
        Returns:
            FAISS index object
        """
        print("\nBuilding FAISS index...")
        
        # Normalize embeddings for cosine similarity (using inner product)
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index (Inner Product = cosine similarity after normalization)
        index = faiss.IndexFlatIP(self.embedding_dim)
        index.add(embeddings)
        
        print(f"✓ FAISS index built with {index.ntotal} vectors")
        return index
    
    def save_index(self, index: faiss.Index, chunks: List[DocumentChunk], 
                   output_dir: str):
        """
        Save FAISS index and metadata to disk
        
        Args:
            index: FAISS index
            chunks: List of DocumentChunk objects
            output_dir: Directory to save index and metadata
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_file = output_path / "faiss_index.bin"
        faiss.write_index(index, str(index_file))
        print(f"✓ Saved FAISS index to {index_file}")
        
        # Save metadata (chunks info)
        metadata_file = output_path / "metadata.pkl"
        metadata = {
            "chunks": [chunk.to_dict() for chunk in chunks],
            "embedding_model": self.embedding_model_name,
            "embedding_dim": self.embedding_dim,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "created_at": datetime.now().isoformat(),
            "num_chunks": len(chunks)
        }
        
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"✓ Saved metadata to {metadata_file}")
        
        # Save human-readable JSON summary
        summary_file = output_path / "index_summary.json"
        summary = {
            "embedding_model": self.embedding_model_name,
            "num_chunks": len(chunks),
            "num_documents": len(set(c.filename for c in chunks)),
            "created_at": datetime.now().isoformat(),
            "documents": {}
        }
        
        # Group chunks by document
        for chunk in chunks:
            if chunk.filename not in summary["documents"]:
                summary["documents"][chunk.filename] = {
                    "num_chunks": 0,
                    "pages": set()
                }
            summary["documents"][chunk.filename]["num_chunks"] += 1
            summary["documents"][chunk.filename]["pages"].add(chunk.page_num)
        
        # Convert sets to sorted lists for JSON
        for doc in summary["documents"].values():
            doc["pages"] = sorted(list(doc["pages"]))
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved summary to {summary_file}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Ingest PDFs and build FAISS vector index for RAG"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../../data",
        help="Directory containing PDF files (default: ../../data)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./index",
        help="Output directory for index and metadata (default: ./index)"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Sentence transformer model name (default: from .env or paraphrase-multilingual-mpnet-base-v2)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=700,
        help="Maximum characters per chunk (default: 700)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="Overlap between chunks in characters (default: 100)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  Shankh.ai PDF Ingestion Pipeline")
    print("=" * 70)
    
    try:
        # Initialize pipeline
        pipeline = PDFIngestionPipeline(
            embedding_model=args.embedding_model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        
        # Process PDFs
        chunks = pipeline.process_pdfs(args.data_dir)
        
        if not chunks:
            print("\n✗ No chunks created. Check PDF files and extraction.")
            return 1
        
        # Generate embeddings
        embeddings = pipeline.create_embeddings(chunks)
        
        # Build FAISS index
        index = pipeline.build_faiss_index(embeddings)
        
        # Save everything
        pipeline.save_index(index, chunks, args.output_dir)
        
        print("\n" + "=" * 70)
        print("  ✓ Ingestion Complete!")
        print("=" * 70)
        print(f"  Index location: {args.output_dir}")
        print(f"  Total chunks: {len(chunks)}")
        print(f"  Ready for retrieval queries!")
        print("=" * 70)
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Error during ingestion: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())


# Unit test example usage (run with pytest):
"""
def test_ingestion_sample():
    '''Test ingestion on first page of one PDF'''
    pipeline = PDFIngestionPipeline(chunk_size=500, chunk_overlap=50)
    
    # Extract just first page
    pages = pipeline.extract_text_from_pdf("../../data/151.pdf")
    assert len(pages) > 0, "Should extract at least one page"
    
    # Chunk first page
    chunks = pipeline.chunk_text(pages[0][1], "151.pdf", 1)
    assert len(chunks) > 0, "Should create at least one chunk"
    
    # Generate embeddings
    embeddings = pipeline.create_embeddings(chunks[:3])  # Test with 3 chunks
    assert embeddings.shape[0] == 3, "Should generate 3 embeddings"
    
    # Build index
    index = pipeline.build_faiss_index(embeddings)
    assert index.ntotal == 3, "Index should contain 3 vectors"
    
    # Test retrieval
    query_embedding = pipeline.model.encode(["loan eligibility criteria"])
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, k=2)
    
    print(f"Top match: {chunks[indices[0][0]].excerpt}")
    assert indices.shape[1] == 2, "Should return 2 results"
"""
