import requests
import config_data as config
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


class RerankerService:
    """重排序服务 - 使用DashScope gte-rerank模型对检索结果进行二次精排"""

    def __init__(self):
        self.api_key = config.api_key
        self.model = config.rerank_model
        self.api_url = config.rerank_api_url

    def rerank(self, query: str, documents: list[Document], top_n: int = None) -> list[Document]:
        """
        对检索到的文档进行重排序，按相关性分数从高到低排列
        :param query: 用户查询
        :param documents: 初始检索到的文档列表
        :param top_n: 重排序后保留的文档数量
        :return: 重排序后的文档列表
        """
        if not documents:
            return []

        top_n = top_n or config.rerank_top_n
        top_n = min(top_n, len(documents))

        #提取文档文本内容
        doc_texts = [doc.page_content for doc in documents]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "input": {
                "query": query,
                "documents": doc_texts
            },
            "parameters": {
                "top_n": top_n,
                "return_documents": False
            }
        }

        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()

            #解析rerank结果，按relevance_score降序排列
            reranked_docs = []
            results = result.get("output", {}).get("results", [])

            for item in results:
                idx = item["index"]
                score = item["relevance_score"]
                #应用rerank分数阈值过滤
                if score >= config.rerank_score_threshold:
                    doc = documents[idx]
                    doc.metadata["rerank_score"] = score
                    reranked_docs.append(doc)

            #如果阈值过滤后没有结果，返回top_n原始文档（保底策略）
            if not reranked_docs:
                print("[Rerank] 所有文档低于阈值，使用Top-N保底返回")
                for item in results[:top_n]:
                    idx = item["index"]
                    doc = documents[idx]
                    doc.metadata["rerank_score"] = item["relevance_score"]
                    reranked_docs.append(doc)

            return reranked_docs

        except requests.exceptions.RequestException as e:
            print(f"[Rerank] 网络请求失败，回退到原始检索结果: {e}")
            return documents[:top_n]
        except (KeyError, ValueError) as e:
            print(f"[Rerank] 响应解析失败，回退到原始检索结果: {e}")
            return documents[:top_n]


class MultiQueryService:
    """多查询生成服务 - 使用LLM生成多个查询变体以提高召回率"""

    def __init__(self, llm):
        self.llm = llm
        self.count = config.multi_query_count

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个查询扩展助手。请根据用户的问题，生成{n}个不同角度的表述方式，"
                       "以便从知识库中检索到更全面的信息。改写后的查询应该保持原意，"
                       "但使用不同的关键词和表述方式，覆盖不同的语义角度。"),
            ("user", "用户问题：{query}\n\n请直接输出{n}个改写后的查询，每行一个，不要有序号和额外说明。")
        ])

        self.chain = self.prompt | self.llm | StrOutputParser()

    def generate_queries(self, query: str) -> list[str]:
        """
        根据原始查询生成多个查询变体（包含原始查询）
        :param query: 原始用户查询
        :return: 查询变体列表（含原始查询，已去重）
        """
        try:
            result = self.chain.invoke({"query": query, "n": self.count})
            #按行分割LLM输出
            queries = [q.strip() for q in result.strip().split("\n") if q.strip()]
            #去除可能的前缀序号（如"1."、"2、"等）
            queries = [q.lstrip("0123456789.、-）) ").strip() for q in queries]
            #过滤无效查询
            queries = [q for q in queries if 0 < len(q) <= 200]

            #将原始查询加入列表首位
            all_queries = [query] + queries[:self.count]

            #去重（保持顺序）
            seen = set()
            unique_queries = []
            for q in all_queries:
                if q not in seen:
                    seen.add(q)
                    unique_queries.append(q)

            print(f"[MultiQuery] 原始查询: {query}")
            print(f"[MultiQuery] 生成变体: {unique_queries[1:]}")
            return unique_queries

        except Exception as e:
            print(f"[MultiQuery] 多查询生成失败，使用原始查询: {e}")
            return [query]


class RetrievalOrchestrator:
    """检索编排器 - 整合多查询生成 + 向量召回 + 去重合并 + 重排序"""

    def __init__(self, llm, vector_store):
        self.llm = llm
        self.vector_store = vector_store
        self.reranker = RerankerService()
        self.multi_query = MultiQueryService(llm) if config.multi_query_enabled else None

    def retrieve(self, query: str) -> list[Document]:
        """
        执行完整的检索流程：多查询生成 → 向量召回 → 去重合并 → 重排序
        :param query: 用户查询
        :return: 检索并重排序后的文档列表
        """
        #1.多查询生成
        if self.multi_query:
            queries = self.multi_query.generate_queries(query)
        else:
            queries = [query]

        #2.向量召回 - 对每个查询执行检索并合并结果
        all_docs = []
        seen_contents = set()

        for q in queries:
            docs = self.vector_store.get_retriever().invoke(q)
            for doc in docs:
                #基于文档内容前100字符做去重
                content_key = doc.page_content[:100]
                if content_key not in seen_contents:
                    seen_contents.add(content_key)
                    all_docs.append(doc)

        print(f"[Retrieval] 多查询召回文档数（去重后）: {len(all_docs)}")

        #3.重排序精排
        if all_docs:
            reranked_docs = self.reranker.rerank(query, all_docs)
            print(f"[Retrieval] 重排序后文档数: {len(reranked_docs)}")
            for doc in reranked_docs:
                score = doc.metadata.get("rerank_score", 0)
                print(f"  - rerank_score: {score:.4f} | {doc.page_content[:50]}...")
            return reranked_docs

        return []


if __name__ == '__main__':
    #快速测试RerankerService
    reranker = RerankerService()
    test_docs = [
        Document(page_content="身高155-165cm，体重75-95斤，建议尺码S。"),
        Document(page_content="真丝材质建议干洗，手洗用真丝专用中性洗涤剂。"),
        Document(page_content="身高170-178cm，体重130-150斤，建议尺码XL。"),
    ]
    result = reranker.rerank("我身高175应该穿什么尺码", test_docs, top_n=2)
    for doc in result:
        print(f"score={doc.metadata.get('rerank_score', 'N/A'):.4f} | {doc.page_content}")
