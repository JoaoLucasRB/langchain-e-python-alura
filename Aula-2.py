from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
# from pydantic import Field, BaseModel
from dotenv import load_dotenv
# from langchain.globals import set_debug
import os

load_dotenv()
base_url = os.getenv("BASE_URL")

model = ChatOpenAI(
    base_url=base_url,
    temperature=0.5,
    api_key="lm-studio"
)

prompt_suggestion = ChatPromptTemplate.from_messages(
    [
        ("system", "Você é um guia de viagens especializado em destinos brasileiros."),
        ("placeholder", "{historico}"),
        ("human", "{query}")
    ]
)

chain = prompt_suggestion | model | StrOutputParser()

memoria = {}
session = "aula_langchain_alura"

def historico_por_sessao(sessao: str):
    if session not in memoria:
        memoria[sessao] = InMemoryChatMessageHistory()
    return memoria[sessao]
    

with_message_history = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=historico_por_sessao,
    input_messages_key="query",
    history_messages_key="historico"
)

lista_de_perguntas = [
    "Quero visitiar um lugar no Braisil, famoso por praias e cultura. Pode sugerir?",
    "Qual a melhor época do ano para ir?",
    "Como chegar até lá? Quais os custos aproximados?"
]

for pergunta in lista_de_perguntas:
    resposta = with_message_history.invoke(
        {
            "query": pergunta,
        },
        config={
            "session_id": session
        }
    )
    print("Usuario: ", pergunta)
    print("IA: ", resposta, "\n")