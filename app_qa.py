import time
from rag import RAGService
import config_data as config
import streamlit as st

#设置标题
st.title("智能客服")
st.divider()       #分隔符

if "message" not in st.session_state:
    st.session_state["message"]=[{"role":"assistant","content":"你好，我是一个电商系统客服，请问有什么可以帮助您？"}]

if "rag" not in st.session_state:
    st.session_state["rag"]=RAGService()

for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])
# 在页面最下方提供用户输入
prompt=st.chat_input()

if prompt:
    #在页面输出用户的提问
    st.chat_message("user").write(prompt)
    #将用户输入存入历史对话
    st.session_state["message"].append({"role":"user","content":prompt})

    ai_res_list=[]
    with st.spinner("AI思考中..."):
        #调用Rag获取AI回答
        res_stream=st.session_state["rag"].chain.stream({"input":prompt},config.session_config)

        # 流式输出
        def capture(generator,cache_list):
            for chunk in generator:
                cache_list.append(chunk)
                yield chunk
            # time.sleep(1.5)
        #将模型返回的流式结果和列表传入capture，
        # 通过yield生成器持续输出打印在web页面
        st.chat_message("assistant").write_stream(capture(res_stream,ai_res_list))
        #将ai回答信息加入历史会话
        st.session_state["message"].append({"role":"assistant","content":"".join(ai_res_list)})


