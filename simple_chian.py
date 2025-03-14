from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-2.0-flash")

parser = StrOutputParser()

temp = PromptTemplate(
    template="Do you know about {topic} explain in a simple way ",
    input_variables=["topic"]
)

chain = temp | model | parser

res = chain.invoke({"topic":"jewar international airport"})

print(res)

chain.get_graph().print_ascii()