from llama_index import Settings
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.dashscope import DashScopeEmbedding

def init_qwen(api_key: str):
    """
    初始化Qwen API设置
    
    Args:
        api_key: DashScope API密钥
    """
    Settings.llm = OpenAILike(
        model="qwen-plus",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=api_key,
        is_chat_model=True
    )
    Settings.embed_model = DashScopeEmbedding(
        model_name="text-embedding-v2",
        api_key=api_key
    ) 