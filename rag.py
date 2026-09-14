#rag核心模块
from langchain.chat_models import init_chat_model
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory

from vector_stores import VectorStoreService
from file_history_store import get_history
from rerank import RetrievalOrchestrator

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

        # 【辅助模型：关闭stream！专门给多查询、子任务同步调用】
        self.llm_sync = init_chat_model(
            model="qwen3-max",
            api_key=config.api_key,
            model_provider="openai",
            base_url=config.base_url,
        )

        #创建向量存储服务实例
        self.vector_store=VectorStoreService(embedding=DashScopeEmbeddings(model=config.embedding_model))
        #创建检索编排器（多查询生成 + 向量召回 + 重排序）
        self.retrieval_orchestrator=RetrievalOrchestrator(self.llm_sync,self.vector_store)

        #拼接提示词
        self.prompt=ChatPromptTemplate.from_messages(
            [
                ("system","请根据以下的参考材料帮我分析尺码选择：\n{context}"),
                MessagesPlaceholder("history"),
                ("human","用户问题：\n{input}")
            ]
        )
        # 构建对话链：内部只做LLM生成，【不再做检索】
        inner_chain = self.prompt | print_prompt | self.llm | StrOutputParser()

        self.conversation_chain=RunnableWithMessageHistory(
            inner_chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history"
        )

    def format_documents(self,docs:list[Document])->str:
        if not docs:
            return "无参考资料"
        formated_str=""
        for doc in docs:
            formated_str+=f"文档片段：{doc.page_content}\n,文档元数据：{doc.metadata}\n\n"
        return formated_str

    def stream_answer(self, user_query:str, session_id:str):
        """对外暴露流式接口：检索放在外面！"""
        # ========= 检索在链外执行，拿到纯字符串query，不会碰到Message对象 =========
        docs = self.retrieval_orchestrator.retrieve(user_query)
        context = self.format_documents(docs)

        session_config=config.session_config
        # 传入上下文+用户提问给对话链
        input_payload = {
            "input": user_query,
            "context": context
        }
        stream=self.conversation_chain.stream(input_payload, session_config)
        # TextAccessor 直接转字符串，兼容新版本langchain
        for chunk in stream:
            text = str(chunk)
            if text is not None and len(text) > 0:
                yield text


if __name__ == '__main__':
    rag=RAGService()
    result = rag.stream_answer("秋天应该穿什么系的衣服比较搭配", session_id="user001")
    for trunk in result:
        print(trunk, end="")