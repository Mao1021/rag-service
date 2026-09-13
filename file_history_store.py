import os,json
from typing import Sequence

from langchain_core.messages import message_to_dict, messages_from_dict, BaseMessage
from langchain_core.chat_history import BaseChatMessageHistory





def get_history(session_id):
    return FileChatMessageHistory(session_id,"./chat_history")

class FileChatMessageHistory(BaseChatMessageHistory):
    def __init__(self,session_id,storage_path):
        self.session_id=session_id          #会话ID
        self.storage_path=storage_path      #不同会话ID的存储路径所在的文件夹路径

        # 完整的文件路径
        self.file_path=os.path.join(self.storage_path,self.session_id)

        #确保文件夹路径存在
        os.makedirs(os.path.dirname(self.file_path),exist_ok=True)


    def add_messages(self,messages:Sequence[BaseMessage])->None:
        #self.messages存储是已有消息
        all_messages=list(self.messages)        #已有的消息列表
        all_messages.extend(messages)           #将新消息加入到已有消息列表

        # 转化为字典存储到文件中
        new_messages=[message_to_dict(message) for message in all_messages]
        with open(self.file_path,'w',encoding='utf-8') as file:
            json.dump(new_messages,file,ensure_ascii=False,indent=2)

    @property           #通过装饰器将messages方法变为成员 属性
    def messages(self)->list[BaseMessage]:
        #当前文件内：list[字典]
        try:
            with open(self.file_path,'r',encoding='utf-8') as file:
                messages_data=json.load(file)      #返回值结果list[字典]
                # 字典转消息对象
                return messages_from_dict(messages_data)
        except FileNotFoundError:
            return []

    def clear(self) -> None:
        with open(self.file_path,'w',encoding='utf-8') as file:
            json.dump([],file,ensure_ascii=False,indent=2)


