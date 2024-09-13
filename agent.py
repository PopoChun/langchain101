
from dotenv import load_dotenv
from langchain.agents import load_tools
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent
from langchain import hub
from langchain.agents import AgentExecutor

load_dotenv()

gpt_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

tools = load_tools(["serpapi", "llm-math"], llm=gpt_llm)

# check tool's description
# print(tools[1].name), print(tools[1].description)

prompt = hub.pull("hwchase17/openai-functions-agent")
prompt.messages

my_agent = create_tool_calling_agent(gpt_llm, tools, prompt)

agent_executor = AgentExecutor(agent=my_agent, tools=tools, verbose=True)

agent_executor.invoke({"input": "誰是鐵達尼號的男主角?"}) # 使用 serpapi tool

agent_executor.invoke({"input": "鐵達尼號的男主角目前幾歲?"}) # 使用 serpapi tool

agent_executor.invoke({"input": "鐵達尼號的男主角十年後幾歲?"}) # 使用 calculator tool
