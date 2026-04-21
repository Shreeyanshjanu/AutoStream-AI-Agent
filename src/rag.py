"""
RAG (Retrieval-Augmented Generation) Pipeline for AutoStream
Handles knowledge base retrieval and answer generation using Gemini
"""

import os
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

genai.configure(api_key=GOOGLE_API_KEY)


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline for AutoStream knowledge base
    """
    
    def __init__(self, kb_path: str = "data/knowledge_base.md"):
        """
        Initialize RAG pipeline with knowledge base
        
        Args:
            kb_path: Path to knowledge base markdown file
        """
        self.kb_path = kb_path
        self.chunks: List[str] = []
        self.full_content: str = ""
        self.model = genai.GenerativeModel("gemini-2.5-flash")
        
        # Load and prepare knowledge base
        self._load_knowledge_base()
        self._split_into_chunks()
        
    def _load_knowledge_base(self) -> None:
        """Load knowledge base from markdown file"""
        try:
            with open(self.kb_path, "r", encoding="utf-8") as f:
                self.full_content = f.read()
            print(f"✅ Knowledge base loaded from {self.kb_path}")
            print(f"   Total content length: {len(self.full_content)} characters")
        except FileNotFoundError:
            raise FileNotFoundError(f"Knowledge base not found at {self.kb_path}")
    
    def _split_into_chunks(self) -> None:
        """
        Split knowledge base into logical chunks using markdown headers
        Each chunk is a section between headers
        """
        lines = self.full_content.split("\n")
        current_chunk = []
        current_header = "Introduction"
        
        for line in lines:
            # Check if line is a markdown header (starts with # or ##)
            if line.strip().startswith("#"):
                # Save previous chunk if it has content
                if current_chunk and any(current_chunk):
                    chunk_text = "\n".join(current_chunk).strip()
                    if len(chunk_text) > 10:  # Only save chunks with meaningful content
                        self.chunks.append(chunk_text)
                
                # Start new chunk
                current_header = line.strip()
                current_chunk = [line]
            else:
                # Add line to current chunk
                current_chunk.append(line)
        
        # Don't forget the last chunk
        if current_chunk and any(current_chunk):
            chunk_text = "\n".join(current_chunk).strip()
            if len(chunk_text) > 10:
                self.chunks.append(chunk_text)
        
        print(f"✅ Knowledge base split into {len(self.chunks)} chunks")
    
    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract keywords from user query for retrieval

        Args:
            query: User's question

        Returns:
            List of lowercase keywords
        """
        # Remove common words
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "is", "are", "was", "were", "be", "been", "do", "does", "did",
            "have", "has", "had", "what", "when", "where", "why", "how", "which",
            "can", "could", "would", "should", "will", "shall", "may", "might",
            "must", "i", "you", "he", "she", "it", "we", "they", "my", "your",
            "get", "got", "s", "t"
        }

        # Split query into words and filter
        words = query.lower().replace("?", "").replace("!", "").split()
        keywords = [w for w in words if w.isalnum() and w not in stop_words and len(w) > 1]

        return keywords
    
    def _score_chunk(self, query: str, chunk: str, keywords: List[str]) -> float:
        """
        Score a chunk based on keyword matches

        Args:
            query: Original user query
            chunk: Knowledge base chunk
            keywords: Extracted keywords from query

        Returns:
            Relevance score (0-100)
        """
        chunk_lower = chunk.lower()
        score = 0.0

        # Exact phrase match (highest priority)
        if query.lower() in chunk_lower:
            score += 100

        # Check for section headers (priority boost for main sections)
        if "### " in chunk or "## " in chunk:
            score += 20

        # Keyword matches - weighted by importance
        for kw in keywords:
            if kw in chunk_lower:
                # Higher weight for key pricing/plan words
                if kw in ["plan", "price", "cost", "pro", "basic", "premium", "refund", "policy", "support", "billing", "cancel"]:
                    score += 30
                elif kw in ["feature", "video", "resolution", "caption", "export", "available", "hours"]:
                    score += 20
                else:
                    score += 10

        # Prefer chunks with pricing tables or lists
        if "$" in chunk or "|" in chunk or "✅" in chunk or "❌" in chunk:
            score += 25

        # Prefer longer, more detailed chunks
        chunk_word_count = len(chunk.split())
        if chunk_word_count > 100:
            score += 15
        elif chunk_word_count > 50:
            score += 10

        return score
    
    def retrieve_chunks(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve most relevant chunks from knowledge base
        
        Args:
            query: User's question
            top_k: Number of top chunks to return
            
        Returns:
            List of dicts with chunk text and score
        """
        keywords = self._extract_keywords(query)
        
        # Score all chunks
        scored_chunks = []
        for i, chunk in enumerate(self.chunks):
            score = self._score_chunk(query, chunk, keywords)
            if score > 0:  # Only include chunks with matches
                scored_chunks.append({
                    "index": i,
                    "text": chunk,
                    "score": score
                })
        
        # Sort by score descending
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top k
        return scored_chunks[:top_k]
    
    def generate_answer(self, query: str, retrieved_chunks: List[Dict]) -> str:
        """
        Generate answer using Gemini with retrieved context
        
        Args:
            query: User's question
            retrieved_chunks: List of retrieved chunks from KB
            
        Returns:
            Generated answer from Gemini
        """
        # Prepare context from retrieved chunks
        context = "\n\n".join([chunk["text"] for chunk in retrieved_chunks])
        
        # Create prompt that forces Gemini to use only the provided context
        prompt = f"""You are a helpful AutoStream customer support assistant.

IMPORTANT: Answer ONLY based on the following knowledge base information. 
If the information is not in the knowledge base, say "I don't have that information."
Do not use your general knowledge.

KNOWLEDGE BASE:
{context}

CUSTOMER QUESTION:
{query}

ANSWER:"""
        
        # Call Gemini
        response = self.model.generate_content(prompt)
        
        return response.text
    
    def answer_question(self, query: str, verbose: bool = False) -> Dict:
        """
        Full RAG pipeline: Retrieve → Augment → Generate
        
        Args:
            query: User's question
            verbose: If True, show retrieval details
            
        Returns:
            Dict with query, retrieved chunks, and answer
        """
        # Step 1: Retrieve
        retrieved = self.retrieve_chunks(query, top_k=3)
        
        if verbose:
            print(f"\n📋 Query: {query}")
            print(f"📚 Retrieved {len(retrieved)} chunks:")
            for i, chunk_info in enumerate(retrieved, 1):
                preview = chunk_info["text"][:100].replace("\n", " ") + "..."
                print(f"   {i}. [Score: {chunk_info['score']:.1f}] {preview}")
        
        # Handle case where no chunks found
        if not retrieved:
            if verbose:
                print("❌ No matching information found")
            return {
                "query": query,
                "retrieved_chunks": [],
                "answer": "I don't have information about that in our knowledge base. Please contact support for help."
            }
        
        # Step 2 & 3: Augment + Generate
        answer = self.generate_answer(query, retrieved)
        
        if verbose:
            print(f"✅ Answer: {answer}\n")
        
        return {
            "query": query,
            "retrieved_chunks": [{"text": c["text"][:200], "score": c["score"]} for c in retrieved],
            "answer": answer
        }


# Singleton instance
_rag_instance = None

def get_rag_pipeline() -> RAGPipeline:
    """Get or create RAG pipeline instance"""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RAGPipeline()
    return _rag_instance


if __name__ == "__main__":
    # Test the RAG pipeline
    print("=" * 60)
    print("RAG PIPELINE TEST")
    print("=" * 60)
    
    rag = get_rag_pipeline()
    
    # Test queries
    test_queries = [
        "What does the Pro plan cost?",
        "Can I get refunds?",
        "What's included in the Basic plan?",
        "Do you have 24/7 support?",
        "What's the maximum resolution?",
        "How many videos can I edit per month?",
    ]
    
    for query in test_queries:
        result = rag.answer_question(query, verbose=True)