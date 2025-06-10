from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = ChatGoogleGenerativeAI(model = "gemini-2.0-flash")
parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Give a detailed report on {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template= "Give 5 important pointer of this followign text. {text}",
    input_variables=["text"]
)

chain = prompt1 | llm | parser | prompt2 | llm | parser

res = chain.invoke("Cricket")



print(chain.get_graph().draw_ascii())
