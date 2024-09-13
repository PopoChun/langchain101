# Langchain 101

**LLM**

大型語言模型：機器去’學習’透過’訓練’理解’自然語言’。just like 鸚鵡，透過上百萬次的訓練，“補全”問題的答案(非創造,是訓練出來的, 補全下面紅框,問答的過程)

![image](https://github.com/user-attachments/assets/fde2b269-f484-44ff-a26b-62e8f02eb3d2)

- Prompt (提示詞)：你的問題描述

LLM 實例

- GPT
- LLaMa (open source)
- …

LLM 如何定義標準化？

- 格式化的輸出
- 問題的規範(長度之類的)
- 多次的 api 調用 連鎖使用
- Api 調用外部服務
- 標準化的開發
- 切換多種模型
- …

LangChain —> framework

幫助 LLM 開發的一種框架

User -> LangChain -> LLM

LangChain

Ex: 訓練想要的AI模樣 -> 早餐店阿姨

- Model I/O

- Prompt template : 對提示的封裝, 提升 model 輸出答案的精準度
    - 阿姨有什麼推薦的{items}嗎, items = 飲料 or 主餐 or 點心
    - Why needed? 規範模型的功能(收斂答案
    - 可以用來讓模型學會(訓練)處理特定的問題 (few-shot)
    
- Memory Cache
    - Conversation
- Output parsers skip
- Data Connection
    ![image](https://github.com/user-attachments/assets/49f0d44c-e835-4cf9-884f-13ad6599992e)

    - Document loader [ i.e.: pdf file(source) —> PDF Loader —> Documents ; url —> url Loader —> Documents]
    - Document transformers: 將 document 拆分成 blocks by 相同意義的
    - Embedding model: document to vector (轉成數字化表示: openAI, Hugging Face 可達到)
        - “我” —> “1.0 , 5.0,  22.0”
        - “林北” —> “0.9, 4.9, 21.9”
    - Vector Store: 儲存向量資料
    - Retriever: 檢索器 (操作 vector store)
        - DEMO:
            
            ```python
            # Document loader
            # pip3 install pypdf
            from langchain.document_loaders import PyPDFLoader
            loader = PyPDFLoader("sample.pdf")
            
            documents = loader.load()
            
            len(documents)
            
            # Document transformers
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            text_splitter = RecursiveCharacterTextSplitter(
            		chunk_size = 100,
            		chunk_overlap = 20,
            		length_function = len,
            )
            pages = loader.load_and_split(text_splitter=text_splitter)
            
            len(pages)
            
            pages[1]
            
            # Embedding model
            # import os
            # os.environ["GOOGLE_API_KEY"] = getpass("MY_GOOGLE_API")
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            *emb_rst = embeddings.embed_doc*uments([pages[1].page_content])
            len(emb_rst), len(emb_rst[0]) #(docu_length,vector_count)
            emb_rst[0]
            
            # Vector Store
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            from langchain.vectorstores import Chroma
            
            vectorstore = Chroma.from_documents(
            	documents=pages,
            	embedding=embeddings,
            	persist_directory="./chroma_db"
            );
            
            # Retriever
            query = "fun"
            docs = vectorstore.similarity_search(query)
            len(docs)
            docs[0]
            ```
            

- Chains
    - LLM chain
        - ex: “幫我取一個通俗好記的名字” —> chain.predict() —> “小明”
        - single chain 最少一次 LLM call
            - 定義 prompt
            - 定義 LLM
            - 定義 chain
            - 運行 prdict
    - Router chain
        - 同時擁有多 chain 下：[math chain], [naming chain], [translator chain], [xxx chain]
            - “幫我取一個通俗好記的名字” —> [decide which chain should be used] —> [naming chain]
            - “幫我取一個通俗好記的名字” —> [naming_chain.predict()] —> “小明”
        - 最少兩次 LLM call (如上先分類、判斷要用哪個 chain, 再運行)
            - 定義 prompts  (有s, 複數)
            - 定義 LLMs/embeddings
            - 定義 chain (多條 chain 並行, 並選擇其一執行)
            - 運行 prdict
    - Sequential chain
        - “幫我取一個通俗好記的名字” —> [給我三個通俗好記的名字] —> “小明、小華、小美” —>
        - “小明、小華、小美, 選出一個通俗好記的名字” —> chain.predict() —> “小明”
            - 定義 prompts  (有s, 複數)
            - 定義 LLMs/embeddings
            - 定義 chain (多條 chain 串起來, 並依序執行)
            - 運行 prdict
    - Transformation chain
        - 處理 簡化 優化 context
            - “幫我取一個男孩的名字, 同時需要具備好記、有意義、富有正義感, 還需要…., 像什麼什麼….” —> [chain.run()] —>  “幫我取一個通俗好記的男孩名字”
    - Document chain 處理長、多內容文本的 chain
        - stuff documents
            
            ![image](https://github.com/user-attachments/assets/0f8b1a94-3844-42b4-8a1f-1b82d46737dd)

        - refine documents chain
            - 除了 stuff doc., 還潤飾 縮減
            
           ![image](https://github.com/user-attachments/assets/51238777-b656-41b4-8078-f2d84ae7da18)

        - map reduce
        - map rerank
- Agent
    - LLM (思考決策)+ memory + tools (LLM, 執行方法)
    - LLM 是一個黑盒子 善於模仿、改寫，但無法和外部交流
    - reasoning and acting, 一有複雜的問題, 由 LLM 執行 reasoning (可能是複雜問題拆解成數個小問題) , agent 負責 acting
    - tools , 定義 agent 可以使用哪些工具
    - 處理問題時可能會用到上述各種 chain 參與
    - **DEMO**:
    
    ```python
    from langchain.agents import load_tools
    from langchain_openai import ChatOpenAI
    from langchain.agents import create_tool_calling_agent
    from langchain import hub
    from langchain.agents import AgentExecutor
    
    import os
    os.environ['OPENAI_API_KEY'] = ''
    os.environ['SERPAPI_API_KEY'] = ''
    
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
    
    ```
    
- Callback
    - 就是那個 callback…
