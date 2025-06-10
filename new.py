from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

llm = ChatGoogleGenerativeAI(model = "gemini-2.0-flash")