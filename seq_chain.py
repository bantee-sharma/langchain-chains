from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-2.0-flash")

template1 = PromptTemplate(
    template="Generate a details report on {topic}",
    input_variables=["topic"]
)

template2 = PromptTemplate(
    template="Generate a 5 pointer summary on the following \n {text}",
    input_variables=["text"]
)

parser = StrOutputParser()

chain = template1 | model |parser|template2|model|parser

res = chain.invoke({"topic":"Evoluation of AI in india"})

print(res)
