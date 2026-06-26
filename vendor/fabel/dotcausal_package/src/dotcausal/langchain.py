"""
LangChain integration for dotcausal.

Provides a CausalRetriever that uses .causal knowledge graphs
for deterministic, zero-hallucination RAG.

Usage:
    from dotcausal.langchain import CausalRetriever

    retriever = CausalRetriever.from_file("knowledge.causal")
    docs = retriever.invoke("COVID fatigue")

    # With LangChain chain:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    prompt = ChatPromptTemplate.from_template(
        "Answer based on these facts:\\n{context}\\n\\nQuestion: {question}"
    )
    llm = ChatOpenAI()
    chain = {"context": retriever, "question": lambda x: x} | prompt | llm
    response = chain.invoke("What causes fatigue in COVID patients?")

Author: David Tom Foss <dtfoss-dev@proton.me>
License: MIT
"""

from typing import List, Optional, Any, Dict
from pathlib import Path

try:
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.documents import Document
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    BaseRetriever = object  # Fallback for type hints
    Document = None
    CallbackManagerForRetrieverRun = None  # Fallback for type hints

from .io import CausalReader


class CausalRetriever(BaseRetriever if HAS_LANGCHAIN else object):
    """
    LangChain Retriever that uses .causal knowledge graphs.

    Unlike vector-based RAG which returns "similar" text,
    CausalRetriever returns logically connected facts with
    full provenance - zero hallucination guarantee.

    Attributes:
        file_path: Path to the .causal file
        top_k: Maximum number of results to return (default: 10)
        include_inferred: Include inferred triplets (default: True)
        min_confidence: Minimum confidence threshold (default: 0.0)
        search_fields: Fields to search in ["trigger", "mechanism", "outcome", "all"]

    Example:
        >>> retriever = CausalRetriever.from_file("knowledge.causal", top_k=5)
        >>> docs = retriever.invoke("mitochondria")
        >>> for doc in docs:
        ...     print(doc.page_content)
        ...     print(doc.metadata["is_inferred"])
    """

    file_path: str = ""
    top_k: int = 10
    include_inferred: bool = True
    min_confidence: float = 0.0
    search_fields: List[str] = ["all"]

    # Internal
    _reader: Optional[CausalReader] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        file_path: str = "",
        top_k: int = 10,
        include_inferred: bool = True,
        min_confidence: float = 0.0,
        search_fields: Optional[List[str]] = None,
        **kwargs
    ):
        if not HAS_LANGCHAIN:
            raise ImportError(
                "LangChain integration requires langchain-core. "
                "Install with: pip install dotcausal[langchain]"
            )

        super().__init__(**kwargs)
        self.file_path = str(file_path)
        self.top_k = top_k
        self.include_inferred = include_inferred
        self.min_confidence = min_confidence
        self.search_fields = search_fields or ["all"]
        self._reader = None

        if self.file_path:
            self._load_reader()

    def _load_reader(self) -> None:
        """Load the .causal file."""
        if not Path(self.file_path).exists():
            raise FileNotFoundError(f"Causal file not found: {self.file_path}")
        self._reader = CausalReader(self.file_path)

    @classmethod
    def from_file(
        cls,
        file_path: str,
        top_k: int = 10,
        include_inferred: bool = True,
        min_confidence: float = 0.0,
        search_fields: Optional[List[str]] = None,
    ) -> "CausalRetriever":
        """
        Create a CausalRetriever from a .causal file.

        Args:
            file_path: Path to the .causal file
            top_k: Maximum results to return
            include_inferred: Include inferred triplets
            min_confidence: Minimum confidence threshold
            search_fields: Fields to search

        Returns:
            CausalRetriever instance
        """
        return cls(
            file_path=file_path,
            top_k=top_k,
            include_inferred=include_inferred,
            min_confidence=min_confidence,
            search_fields=search_fields,
        )

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """
        Retrieve relevant documents from the knowledge graph.

        This is the core LangChain interface method.
        """
        if self._reader is None:
            raise ValueError("No .causal file loaded. Use from_file() or set file_path.")

        # Search the knowledge graph
        results = self._reader.search(
            query=query,
            limit=self.top_k * 2,  # Get more, filter later
        )

        # Filter by confidence and inferred status
        filtered = []
        for r in results:
            if r.get("confidence", 1.0) < self.min_confidence:
                continue
            if not self.include_inferred and r.get("is_inferred", False):
                continue
            filtered.append(r)
            if len(filtered) >= self.top_k:
                break

        # Convert to LangChain Documents
        documents = []
        for r in filtered:
            # Format as readable text
            tag = "[INFERRED]" if r.get("is_inferred") else "[EXPLICIT]"
            content = (
                f"{tag} {r['trigger']} → {r['mechanism']} → {r['outcome']}"
            )

            # Rich metadata for downstream use
            metadata = {
                "trigger": r["trigger"],
                "mechanism": r["mechanism"],
                "outcome": r["outcome"],
                "confidence": r.get("confidence", 1.0),
                "is_inferred": r.get("is_inferred", False),
                "source": r.get("source", ""),
                "provenance": r.get("provenance", []),
            }

            documents.append(Document(page_content=content, metadata=metadata))

        return documents

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the loaded knowledge graph."""
        if self._reader is None:
            raise ValueError("No .causal file loaded.")
        return self._reader.get_stats()

    def __repr__(self) -> str:
        stats = ""
        if self._reader:
            try:
                s = self._reader.get_stats()
                stats = f", triplets={s.get('explicit_triplets', '?')}+{s.get('inferred_triplets', '?')}"
            except:
                pass
        return f"CausalRetriever(file='{self.file_path}', top_k={self.top_k}{stats})"


# Convenience function for quick setup
def create_causal_retriever(
    file_path: str,
    top_k: int = 10,
    include_inferred: bool = True,
) -> CausalRetriever:
    """
    Quick helper to create a CausalRetriever.

    Example:
        retriever = create_causal_retriever("knowledge.causal")
    """
    return CausalRetriever.from_file(
        file_path=file_path,
        top_k=top_k,
        include_inferred=include_inferred,
    )


__all__ = ["CausalRetriever", "create_causal_retriever", "HAS_LANGCHAIN"]
