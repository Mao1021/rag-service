import os

md5_path="./md5.txt"

api_key=os.getenv("DASHSCOPE_API_KEY")
base_url=os.getenv("DASHSCOPE_BASE_URL")



#Chroma
collection_name="RAG"
persist_directory="./chroma_db"

#RecursiveCharacterTextSplitter
chunk_size=1000
chunk_overlap=100
separators=["\n\n","\n",",",".","!","?","，","。","？","！"]
min_split_chunk_str=1000  #最小分割阈值


#Retriever
top_k=2
similarity_threshold=0.6

#RAG

embedding_model="text-embedding-v4"
chat_model="qwen3-max"


#配置session_id
session_config={
        "configurable":{
            "session_id":"user001"
        }
    }