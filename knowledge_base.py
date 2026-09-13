#离线模块

import os
import config_data as config
import hashlib
from datetime import datetime
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

def check_md5(md5str:str):
    """
    检查传入的md5是否已经被处理过
    :return:
    """
    #如果文件不存在，则表明未处理
    if not os.path.exists(config.md5_path):
        #创建该文件
        open(config.md5_path,'w',encoding='utf-8').close()
        return False
    #如果文件存在，则将文件内容与md5str进行匹对
    else:
        for line in open(config.md5_path,'r',encoding='utf-8').readlines():
            #匹对成功
            if line.strip()==md5str:
                return True
        #匹对失败
        return False

def save_md5(md5str:str):
    """
    将传入的md5字符串，记录到文件内保存
    :return:
    """
    with open(config.md5_path,'a',encoding='utf-8') as f:
        f.write(md5str+"\n")

def get_string_md5(input_str:str,encoding='utf-8'):
    """
    将传入的字符串转换为md5字符串
    :return:
    """
    #将字符串转换为bytes字节数组
    str_bytes=input_str.encode(encoding=encoding)

    # 创建md5对象
    md5_obj=hashlib.md5()           #得到md5对象
    md5_obj.update(str_bytes)       #更新内容（传入即将要转换的字节数组）
    md5_hex=md5_obj.hexdigest()     #得到md5的16进制字符串

    return md5_hex





class KnowledgeBaseService(object):
    def __init__(self):
        #如过文件夹不存在 则创建
        os.makedirs(config.persist_directory,exist_ok=True)
        self.chroma=Chroma(
            collection_name=config.collection_name,
            embedding_function=DashScopeEmbeddings(model="text-embedding-v4"),
            persist_directory=config.persist_directory

        )        #向量存储的实例Chroma向量数据库
        self.spliter=RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators,
            length_function=len
        )       #文本切割器的对象

    def upload_by_str(self,data,filename):
        """
        将传入的字符串向量化并存入向量数据库中
        :param data:
        :param filename:
        :return:
        """
        #将数据转化为16进制
        md5_hex=get_string_md5(data)
        #判断数据是否已经存在MD5文件中
        if check_md5(md5_hex):
            return "[跳过]内容已存在知识库中"
        #判断文本的长度是否大于最小分割阈值
        if len(data)>config.min_split_chunk_str:
            knowledge_chunk:list[str]=self.spliter.split_text(text=data)

        else:
            knowledge_chunk=[data]

        metadata={
            "source":filename,
            "create_time":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator":"小明"
        }
        self.chroma.add_texts(
            texts=knowledge_chunk,
            metadatas=[metadata for _ in knowledge_chunk]
        )

        save_md5(md5_hex)
        return "[成功]内容已经成载入向量库"

if __name__ == '__main__':
    K=KnowledgeBaseService()
    check_md5("7eca689f0d3389d9dea66ae112e5cfd7")
    # print(get_string_md5("你好"))
    # r=K.upload_by_str("笨蛋","p.txt")
    # print(r)
    pass