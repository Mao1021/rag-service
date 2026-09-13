import time

import streamlit as st
from knowledge_base import KnowledgeBaseService

# 添加网页标题
st.title("知识库跟新服务")

# 文件上传框
uploader_file=st.file_uploader(
    label="请上传文件",
    type=['txt'],
    accept_multiple_files=False     #Fasle默认只允许一个文件上传
)

#st.session_state是一个字典 能保存状态
if "service" not in st.session_state:
    st.session_state["service"]=KnowledgeBaseService()




if uploader_file is not None:
    # 提取文件的名字
    file_name=uploader_file.name
    # 提取文件的类型
    file_type=uploader_file.type
    # 计算文件的大小（KB）
    file_size=uploader_file.size/1024

    st.subheader(f"文件名：{file_name}")
    st.write(f"格式：{file_type} | 大小：{file_size:.2f}KB")

    # 获取文件的内容
    text=uploader_file.getvalue().decode('utf-8')


    with st.spinner("数据入库中..."):
        time.sleep(2)
        # 文件入库
        service:KnowledgeBaseService = st.session_state["service"]
        response=service.upload_by_str(text, file_name)
        st.write(response)