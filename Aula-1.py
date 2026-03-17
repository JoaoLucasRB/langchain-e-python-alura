from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import Field, BaseModel
from dotenv import load_dotenv
from langchain.globals import set_debug
import os

set_debug(True)

load_dotenv()
base_url = os.getenv("BASE_URL")

class Destino(BaseModel):
    cidade:str = Field("A cidade recomendada para visitar")
    motivo:str = Field("Motivo pelo qual é interessante visitar essa cidade")
    
class Restaurantes(BaseModel):
    cidade:str = Field("A cidade recomendada para visitar")
    restaurantes:str = Field("Restaurantes recomendados na cidade")

parser_destino = JsonOutputParser(pydantic_object=Destino)
parser_restaurantes = JsonOutputParser(pydantic_object=Restaurantes)

prompt_city = PromptTemplate(
    template="""
    Sugira uma cidade dado o meu interesse por {interesse}
    {formato_resposta}
    """,
    input_variables=["interesse"],
    partial_variables={"formato_resposta": parser_destino.get_format_instructions()}
)

prompt_restaurants = PromptTemplate(
    template="""
    Sugira restaurantes populares entre locais em {cidade}
    {formato_resposta}
    """,
    partial_variables={"formato_resposta": parser_restaurantes.get_format_instructions()}
)

prompt_culture = PromptTemplate(
    template="""
        Sugira atividades culturais em {cidade}
    """
)

model = ChatOpenAI(
    base_url=base_url,
    temperature=0.5,
    api_key="lm-studio"
)

chain_city = prompt_city | model | parser_destino
chain_restaurants = prompt_restaurants | model | parser_restaurantes
chain_culture = prompt_culture | model | StrOutputParser()

chain = (
    chain_city | chain_restaurants | chain_culture
)

response = chain.invoke(
    {
        "interesse": "praias"
    }
)

print(response)