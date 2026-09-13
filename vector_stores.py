import config_data as config
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma


class VectorStoreService(object):
    def __init__(self,embedding):
        self.embedding=embedding

        self.vector_store=Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory
        )

    #获取检索器
    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k":config.top_k})

if __name__ == '__main__':
    v=VectorStoreService(DashScopeEmbeddings(model="text-embedding-v4")).get_retriever()
    result=v.invoke("我身高155")
    print(result)
    # for content in result:
    #     print(content)