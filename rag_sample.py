import bs4
import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    ConfigurableField,
    RunnablePassthrough,
)
from langchain_core.runnables import RunnableParallel

load_dotenv()
os.getenv('GOOGLE_API_KEY')

llm = OllamaFunctions(
    model="llama3:8b"
)  # assuming you have Ollama installed and have llama3 model pulled with `ollama pull llama3 `

# print("BEFORE:")
# print(llm.invoke("Who is the current Chairman of the Department of Information Engineering in Feng Chia University?"))

loader = WebBaseLoader("https://www.iecs.fcu.edu.tw/en/teacher/#major")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma.from_documents(
	documents=splits,
	embedding=embeddings,
	persist_directory="./chroma_db"
);

retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# print("AFTER:")
# print(rag_chain.invoke("Who is the current Chairman of the Department of Information Engineering in Feng Chia University?"))
# rag_chain.invoke("Why is apple not a fruit?")

class GradeHallucinations(BaseModel):
    """
    確認答案是否為虛構(yes/no)
    """
    binary_score: str = Field(description="答案是否由為虛構。('yes' or 'no')")

# Prompt Template
instruction = """
你是一個評分的人員，負責確認LLM的回應是否為虛構的。
以下會給你一個文件與相對應的LLM回應，請輸出 'yes' or 'no'做為判斷結果。
'Yes' 代表LLM的回答是虛構的，未基於文件內容 'No' 則代表LLM的回答並未虛構，而是基於文件內容得出。
"""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",instruction),
        ("human", "文件: \n\n {documents}"),
    ]
)

# Grader LLM
# grader_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

# 使用 LCEL 語法建立 chain
hallucination_grader = hallucination_prompt | structured_llm_grader

class GradeAnswer(BaseModel):
    """
    確認答案是否可回應問題
    """

    binary_score: str = Field(description="答案是否回應問題。('yes' or 'no')")

# Prompt Template
instruction = """
你是一個評分的人員，負責確認答案是否回應了問題。
輸出 'yes' or 'no'。 'Yes' 代表答案確實回應了問題， 'No' 則代表答案並未回應問題。
"""
# Prompt
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",instruction),
        ("human", "使用者問題: \n\n {question} \n\n 答案: {generation}"),
    ]
)

# LLM with function call
# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeAnswer)

# 使用 LCEL 語法建立 chain
answer_grader = answer_prompt | structured_llm_grader

result = llm.invoke("Who is the current Chairman of the Department of Information Engineering in Feng Chia University?")
score = hallucination_grader.invoke({"documents": result})
grade = score.binary_score


print(grade)
    
