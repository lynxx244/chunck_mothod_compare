# 文本切分实验全局配置
@dataclass
class Config:
    # 文本切分基础参数
    CHUNK_SIZE = 500
    OVERLAP = 50
    MAX_CHUNK = 850
    MIN_CHUNK = 100
    THRESHOLD = 0.4

    # 检索评估参数
    TOP_K = 3
    QA_COUNT = 80

    # 文件路径
    FILE_PATH = "内科学第10版.txt"
    TEXT_LIMIT = 150000

    # 模型配置
    EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
