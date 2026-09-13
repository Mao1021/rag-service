from langchain.chat_models import init_chat_model
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory, RunnableLambda

from vector_stores import VectorStoreService
from file_history_store import get_history

import config_data as config

def print_prompt(prompt):
    print("="*20)
    print(prompt.to_string())
    print("="*20)
    return prompt

class RAGService(object):

    def __init__(self):
        # 创建大模型
        self.llm=init_chat_model(
            model="qwen3-max",
            api_key=config.api_key,
            model_provider="openai",
            base_url=config.base_url,
            stream=True
        )

        #创建向量存储服务实例
        self.vector_store=VectorStoreService(embedding=DashScopeEmbeddings(model=config.embedding_model))
        #拼接提示词
        self.prompt=ChatPromptTemplate.from_messages(
            [
                ("system","请根据以下的参考材料帮我分析尺码选择：\n{context}"),
                ("system","并且我提供用户的对话历史记录，如下："),
                MessagesPlaceholder("history"),
                ("user","用户问题：\n{input}")
            ]
        )
        #构造chain链
        self.chain=self.__get_chain()


    def __get_chain(self):

        retriever=self.vector_store.get_retriever()

        def format_documents(docs:list[Document]):
            if not docs:
                return "无参考资料"
            formated_str=""
            for doc in docs:
                formated_str+=f"文档片段：{doc.page_content}\n,文档元数据：{doc.metadata}\n\n"

            return formated_str


        def format_for_retriever(value:dict)->str:
            return value['input']
        def format_for_prompt(value:dict)->dict:
            new_dict={}
            new_dict['input']=value['input']['input']
            new_dict['history']=value['input']['history']
            new_dict['context']=value['context']
            return new_dict

        chain=(
            {
            "input":RunnablePassthrough(),
            "context":RunnableLambda(format_for_retriever)|retriever|format_documents
            }
            |RunnableLambda(format_for_prompt)|self.prompt|print_prompt|self.llm|StrOutputParser()
        )

        conversation_chain=RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history"
        )



        return conversation_chain

if __name__ == '__main__':
    #配置session_id
    session_config={
        "configurable":{
            "session_id":"user001"
        }
    }
    rag=RAGService()
    result=rag.chain.invoke({"input":"秋天应该穿什么系的衣服比较搭配"},session_config)
    print(result)