# rag.py
# --- CAMBIOS DE IMPORTACIÓN ---
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings # Usando la solución local anterior
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
# ------------------------------
from typing import List

# Documentos de ejemplo para RAG
SAMPLE_DOCUMENTS = [
    "Política de Cancelación: Se aceptan cancelaciones hasta 48 horas antes de la llegada con reembolso total.",
    "Manual de usuario del agente: El agente solo puede reservar glamping, no hoteles.",
    "Horario de Check-in: El check-in es a las 15:00 horas. El check-out es a las 11:00 horas."
]

class RAGSystem:
    def __init__(self):
        # ADAPTAR: Usar el servicio de Embeddings de tu proveedor de Llama 3 (o uno compatible)
        self.embeddings = HuggingFaceEmbeddings()
        self.vectorstore = self._create_vectorstore()

    def _create_vectorstore(self):
        """Crea e indexa el vector store (simulación)."""
        docs = [Document(page_content=d) for d in SAMPLE_DOCUMENTS]
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        split_docs = text_splitter.split_documents(docs)
        
        # Indexación
        vectorstore = FAISS.from_documents(split_docs, self.embeddings)
        print("✅ RAG: Documentos indexados correctamente.")
        return vectorstore

    def retrieve_context(self, query: str, k: int = 1) -> str:
        """Recupera el contexto más relevante para una query."""
        retrieved_docs = self.vectorstore.similarity_search(query, k=k)
        context = "\n".join([doc.page_content for doc in retrieved_docs])
        return context

# Instancia global
rag_system = RAGSystem()