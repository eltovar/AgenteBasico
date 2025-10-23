# llm_client.py
from langchain_ollama import ChatOllama # Cambio a la importación oficial
from langchain_core.messages import BaseMessage # Tipificación estricta
from typing import List
from dotenv import load_dotenv
import os

load_dotenv()

LLM_MODEL_NAME = "gpt-5-nano"

class LLMClient:
    def __init__(self, model_name: str = LLM_MODEL_NAME, temperature: float = 0.1):
        # ADAPTACIÓN: Usamos el cliente Ollama que se conecta a tu instancia local
        self.client = ChatOllama(
            model=model_name, 
            temperature=temperature,
            # Asegúrate de que Ollama esté corriendo en segundo plano
        )

    # Tipificación mejorada: List[BaseMessage] indica que debe ser una lista
    # de mensajes de LangChain.
    def get_response(self, messages: List[BaseMessage]) -> str:
        """Obtiene una respuesta de la LLM."""
        try:
            # El cliente Ollama usa el mismo método de invocación
            response = self.client.invoke(messages)
            return response.content
        except Exception as e:
            print(f"Error en LLMClient con Ollama: {e}")
            return "Lo siento, hubo un error al procesar tu solicitud."

# Instancia global
llama_client = LLMClient()