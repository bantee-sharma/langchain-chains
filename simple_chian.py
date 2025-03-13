from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-2.0-flash")

tempalte = PromptTemplate(
    template= "Give me  5 intresting facts about {topic} in one-one line and assign number before strating a new line",
    input_variables=["topic"]
)

parser = StrOutputParser()

chain = tempalte | model | parser

res = chain.invoke({"topic":"UPSC"})

print(res)

chain.get_graph().print_ascii()