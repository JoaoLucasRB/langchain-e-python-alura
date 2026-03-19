from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()
base_url = os.getenv("BASE_URL")

model = ChatOpenAI(
    model="google/gemma-3n-e4b",
    base_url=base_url,
    temperature=0.5,
    api_key="lm-studio"
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

arquivos = [
    "documentos/GTB_standard_Nov23.pdf",
    "documentos/GTB_gold_Nov23.pdf",
    "documentos/GTB_platinum_Nov23.pdf"
]

documentos = sum([PyPDFLoader(arquivo).load() for arquivo in arquivos], [])

pedacos = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
).split_documents(documentos)

dados_recuperados = FAISS.from_documents(pedacos, embeddings).as_retriever(search_kwargs={"k": 2})

prompt_consulta_seguro = ChatPromptTemplate.from_messages(
    [   
        ("system", "Responda usando exclusivamente o conteúdo fornecido."),
        ("human", "{query}\n\nContexto:\n{contexto}\n\nResposta:")
    ] 
)

cadeia = prompt_consulta_seguro | model | StrOutputParser()

def responder(pergunta:str):
    trechos = dados_recuperados.invoke(pergunta)
    contexto = "\n\n".join([trecho.page_content for trecho in trechos])
    return cadeia.invoke({"query": pergunta, "contexto": contexto})

print(responder("Como devo proceder caso tenha um item comprado roubado e caso eu tenha o cartão gold?"))