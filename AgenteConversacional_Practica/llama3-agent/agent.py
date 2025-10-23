# agent.py (OPTIMIZADO)
from llm_client import llama_client
from rag import rag_system
from tool import ALL_TOOLS, informacion_empresa_func
from prompts.tool_prompts import SYSTEM_AGENT_PROMPT, TOOL_DECISION_PROMPT
from langchain_core.messages import SystemMessage, HumanMessage
import json
import re

class ConversationalAgent:
    def __init__(self, tools=ALL_TOOLS):

        self.tools = {tool.name: tool for tool in tools}
        self.tool_descriptions = "\n".join([f"- {t.name}: {t.description}" for t in tools])

    def _run_tool(self, tool_name: str, tool_input: dict) -> str:
        """Ejecuta una herramienta y devuelve la respuesta."""

        if tool_name == "info_empresa_contacto_filosofia":
            # Ejecución explícita del func con sus argumentos desempaquetados
            return informacion_empresa_func(**tool_input)

        return "Tool no encontrada."

    def _determine_tool_call(self, user_input: str) -> dict | None: # Anotación de retorno mejorada
        """
        Determina si la consulta del usuario requiere una Tool.
        Retorna un diccionario con 'tool_name' y 'tool_input' si aplica, o None si no.
        """

        # 1. Construir la instrucción de prompting CLARA
        full_instruction_prompt = (
            # Usamos TOOL_DECISION_PROMPT como base (debería ser conciso)
            TOOL_DECISION_PROMPT.format(user_input=user_input) +
            f"\n\nHerramientas disponibles:\n{self.tool_descriptions}" +
            # Instrucción crítica de formato al final
            "\n\nResponde ÚNICAMENTE con un JSON válido en el formato: "
            '{"tool_name": "nombre_tool", "tool_input": {"param1": "valor1", "param2": "valor2"}}. '
            "Si no se requiere ninguna herramienta, responde ÚNICAMENTE: NO_TOOL"
        )
        
        messages = [
            SystemMessage(content=SYSTEM_AGENT_PROMPT),
            HumanMessage(content=full_instruction_prompt) # Enviamos la instrucción completa
        ]

        response = llama_client.get_response(messages)

        # 2. Parsear y manejar errores
        response_clean = response.strip()

        if "NO_TOOL" in response_clean.upper():
            return None

        try:
            # Buscar el primer bloque JSON (más robusto contra texto envolvente)
            json_match = re.search(r'\{.*\}', response_clean, re.DOTALL)
            if json_match:
                tool_call = json.loads(json_match.group(0))
                # Validar estructura básica del JSON
                if 'tool_name' in tool_call and 'tool_input' in tool_call:
                    return tool_call

                print(f"⚠️ Warning: JSON encontrado pero estructura incorrecta: {tool_call}")
                return None
            else:
                return None
        except json.JSONDecodeError:
            print(f"⚠️ Error al parsear JSON de Tool. Respuesta LLM: {response_clean[:50]}...")
            return None

    def process_query(self, user_input: str) -> str:
        """Procesa la consulta del usuario, usando el flujo de decisión."""
        print(f"\nUsuario: {user_input}")

        # 1. Detección de Tools (Único camino para RAG/Información de la empresa)
        tool_call = self._determine_tool_call(user_input)

        if tool_call and 'tool_name' in tool_call and 'tool_input' in tool_call:
            tool_name = tool_call['tool_name']
            tool_input = tool_call['tool_input']

            print(f"🤖 Agente (Tool): Ejecutando Tool '{tool_name}' con input: {tool_input}")

            # Solo manejamos la tool de info_empresa (que lleva a RAG)
            if tool_name == "info_empresa_contacto_filosofia":
                print("🤖 Agente (RAG): Pasando a recuperar contexto de RAG tras Tool-Call...")

                # Usamos el 'tema' (el input de la Tool) como la query de RAG
                # Si 'tema' no está presente (por fallo de LLM), usamos el input original
                rag_query = tool_input.get('tema', user_input)
                context = rag_system.retrieve_context(rag_query)

                rag_prompt = (
                    f"{SYSTEM_AGENT_PROMPT}\n\n"
                    f"Tu herramienta de info_empresa_contacto_filosofia te ha dirigido a usar RAG. "
                    f"Usa el siguiente contexto para responder a la pregunta original del usuario: '{user_input}'.\n"
                    f"Contexto: {context}"
                )

                messages = [
                    SystemMessage(content=rag_prompt),
                    HumanMessage(content=user_input)
                ]

                response = llama_client.get_response(messages)
                return f"💬 Agente (RAG, impulsado por Tool): {response}"

            # Si fuera otra Tool (hypotética), ejecutaría aquí
            tool_response = self._run_tool(tool_name, tool_input)
            return f"✅ Respuesta de la Tool ({tool_name}): {tool_response}"


        # 2. LLM Base (Respuesta conversacional general)
        print("🤖 Agente (LLM): Respondiendo directamente...")
        messages = [
            SystemMessage(content=SYSTEM_AGENT_PROMPT),
            HumanMessage(content=user_input)
        ]
        response = llama_client.get_response(messages)
        return f"💡 Agente (LLM): {response}"

# Instancia global
agent = ConversationalAgent()