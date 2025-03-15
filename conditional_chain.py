from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Literal
from langchain_core.runnables import RunnableBranch,RunnableLambda

load_dotenv()


model = ChatGoogleGenerativeAI(model = "gemini-2.0-flash")

parser = StrOutputParser()

class feedback(BaseModel):
    sentiment : Literal["positive", "negative"] = Field(description="Give the sentiment of the feedback")

parser2 = PydanticOutputParser(pydantic_object=feedback)

template1 = PromptTemplate(
    template="Classify the given feedback as positive or negative \n {feedback} \n {format_instruction}",
    input_variables=["feedback"],
    partial_variables={"format_instruction": parser2.get_format_instructions()}
)

classifier_chain = template1 | model | parser2

prompt2 = PromptTemplate(
    template="Write a 2 line appropriate response to this positive feedback \n {feedback}",
    input_variables=["feedback"]
)

prompt3 = PromptTemplate(
    template="Write a 2 line appropriate response to this negative feedback \n {feedback}",
    input_variables=["feedback"]
)

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", prompt2 | model | parser),
    (lambda x: x.sentiment == "negative", prompt3 | model |parser),
    RunnableLambda(lambda x: "could not find sentiment")
)

chain = classifier_chain | branch_chain

res = chain.invoke({'feedback':"this phone has very good features so i like this phone "})

print(res)



