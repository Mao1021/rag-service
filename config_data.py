# 系统配置文件模块

import os

md5_path="./md5.txt"

api_key=os.getenv("DASHSCOPE_API_KEY")
base_url=os.getenv("DASHSCOPE_BASE_URL")



#Chroma
collection_name="RAG"
persist_directory="./chroma_db"

#RecursiveCharacterTextSplitter
chunk_size=400  #每个分块约400字符，聚焦单个主题（如一种材质的洗护方法）
chunk_overlap=50  #较小overlap即可保留上下文，避免内容重复率过高
separators=["\n\n","\n","。","，","！","？",".",",","!","?"]  #优先按段落→行→中文标点切分
min_split_chunk_str=400  #与chunk_size一致，短于此值的文本不切分


#Retriever
top_k=3  #知识库总量小，3篇已覆盖大部分相关内容，配合rerank精排
similarity_threshold=0.5  #小知识库适当放宽阈值，避免过度过滤导致无结果

#RAG
embedding_model="text-embedding-v4"
chat_model="qwen3-max"

#Rerank（重排序）
rerank_api_url="https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
rerank_model="gte-rerank-v2"  #DashScope重排序模型
rerank_top_n=2  #重排序后最终保留的文档数量
rerank_score_threshold=0.2  #rerank分数阈值，低于此值的文档将被丢弃

#Multi-Query（多查询检索策略）
multi_query_enabled=True  #是否启用多查询生成
multi_query_count=3  #生成的查询变体数量（不含原始查询）


#配置session_id
session_config={
        "configurable":{
            "session_id":"user001"
        }
    }