# ===================== 【终极修复版】文本切分实验代码 =====================
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import re
import numpy as np
import faiss
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass, field
from typing import Dict, List
import torch

# 设备自动检测
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_FP16 = True if DEVICE == "cuda" else False

# ===================== 全局参数 =====================
@dataclass
class Config:
    CHUNK_SIZE = 500
    OVERLAP = 50
    MAX_CHUNK = 450      # 适配BGE-base模型，不超token限制
    MIN_CHUNK = 150
    THRESHOLD = 0.65
    TOP_K = 3
    FILE_PATH = r"C:\Users\lscher\OneDrive\Desktop\大模型实验\文本切割研究\内科学第10版.txt"

# ===================== Chunk 结构 =====================
@dataclass
class Chunk:
    content: str
    metadata: Dict = field(default_factory=dict)
    chunk_id: str = ""

# ===================== 1. 固定长度切分 =====================
class FixedSizeChunker:
    def __init__(self, chunk_size=Config.CHUNK_SIZE, overlap=Config.OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text, source=""):
        chunks = []
        start = 0
        idx = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(Chunk(
                    content=chunk_text,
                    metadata={"method": "fixed"},
                    chunk_id=f"fixed_{idx}"
                ))
                idx += 1
            start = end - self.overlap
        return chunks

# ===================== 2. 递归结构切分 =====================
class RecursiveChunker:
    def __init__(self, chunk_size=Config.CHUNK_SIZE, overlap=Config.OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = ["\n\n", "\n", "。", "！", "？", "；", " "]

    def split(self, text, source=""):
        chunks = self._recurse(text, 0)
        res = []
        for i, c in enumerate(chunks):
            c = c.strip()
            if c:
                res.append(Chunk(content=c, metadata={"method": "recursive"}, chunk_id=f"recur_{i}"))
        return res

    def _recurse(self, text, sep_idx):
        if len(text) <= self.chunk_size or sep_idx >= len(self.separators):
            return [text]

        sep = self.separators[sep_idx]
        splits = text.split(sep)
        chunks = []
        current = ""

        for part in splits:
            part += sep
            if len(current) + len(part) <= self.chunk_size:
                current += part
            else:
                if current:
                    chunks.append(current.strip())
                current = part
                if len(current) > self.chunk_size:
                    sub_chunks = self._recurse(current, sep_idx + 1)
                    chunks.extend(sub_chunks)
                    current = ""

        if current.strip():
            chunks.append(current.strip())
        return chunks

# ===================== 3. 传统相邻语义切分 =====================
class OriginalSemanticChunker:
    def __init__(self, model, threshold=0.55, max_chunk=Config.CHUNK_SIZE, min_chunk=Config.MIN_CHUNK):
        self.model = model
        self.threshold = threshold
        self.max_chunk = max_chunk
        self.min_chunk = min_chunk

    def _split_sentences(self, text):
        pattern = r'(?<=[。！？；\n])'
        parts = re.split(pattern, text)
        return [s.strip() for s in parts if s.strip()]

    def split(self, text, source=""):
        sentences = self._split_sentences(text)
        if not sentences: return []
        emb = self.model.encode(sentences, batch_size=64, show_progress_bar=False)
        chunks = []
        current = [sentences[0]]
        length = len(sentences[0])
        for i in range(1, len(sentences)):
            sim = cosine_similarity([emb[i-1]], [emb[i]])[0][0]
            length += len(sentences[i])
            if sim < self.threshold or length > self.max_chunk:
                chunks.append("".join(current))
                current = [sentences[i]]
                length = len(sentences[i])
            else:
                current.append(sentences[i])
        chunks.append("".join(current))
        merged = []
        buf = ""
        for c in chunks:
            if len(buf) + len(c) < self.min_chunk:
                buf += c
            else:
                if buf:
                    merged.append(buf)
                    buf = c
                merged.append(c)
        if buf:
            merged.append(buf)
        return [Chunk(content=m, metadata={"method": "semantic"}, chunk_id=f"sem_{i}") for i, m in enumerate(merged)]

# ===================== 🔥 自研MaxMin混合切分（彻底修复版） =====================
class RecursiveMaxMinHybridChunker:
    def __init__(self, model, min_chunk=Config.MIN_CHUNK, max_chunk=Config.MAX_CHUNK, threshold=Config.THRESHOLD):
        self.model = model
        self.min_size = min_chunk
        self.max_size = max_chunk
        self.threshold = threshold
        # 修复1：放大基础块，给合并留足空间（和递归切分的基础块大小一致）
        self.base_recursive = RecursiveChunker(chunk_size=400)

    def _split_sentences(self, text):
        # 严格按句子边界切分，保证不截断完整句
        pattern = r'(?<=[。！？；\n])'
        parts = re.split(pattern, text)
        return [s.strip() for s in parts if s.strip()]

    def _local_semantic_merge(self, base_chunks):
        chunk_texts = [c.content for c in base_chunks]
        chunk_embeddings = self.model.encode(chunk_texts, batch_size=64,
                                              show_progress_bar=False)

        merged_result = []
        current_texts = [chunk_texts[0]]
        # 关键改动：用簇内所有块的平均向量，不是最后一个块的向量
        current_mean_emb = chunk_embeddings[0].copy()
        current_count = 1
        current_len = len(chunk_texts[0])

        for i in range(1, len(base_chunks)):
            next_text = chunk_texts[i]
            next_emb = chunk_embeddings[i]

            sim = cosine_similarity([current_mean_emb], [next_emb])[0][0]

            # 双重判断：语义相似 且 长度不超限
            if sim >= self.threshold and current_len + len(next_text) <= self.max_size:
                current_texts.append(next_text)
                current_len += len(next_text)
                # 更新平均向量（增量计算，不重新encode）
                current_mean_emb = (current_mean_emb * current_count + next_emb) / (current_count + 1)
                current_count += 1
            else:
                merged_result.append("".join(current_texts))
                current_texts = [next_text]
                current_mean_emb = next_emb.copy()
                current_count = 1
                current_len = len(next_text)

        if current_texts:
            merged_result.append("".join(current_texts))

        return merged_result

    def _strict_length_normalize(self, chunks):
        result = []
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk: continue

            # 修复3：超过max_size的块，按句子边界切分，绝不截断句子
            if len(chunk) > self.max_size:
                sentences = self._split_sentences(chunk)
                temp_buf = ""
                for sent in sentences:
                    if len(temp_buf) + len(sent) <= self.max_size:
                        temp_buf += sent
                    else:
                        if temp_buf:
                            result.append(temp_buf)
                        temp_buf = sent
                if temp_buf:
                    result.append(temp_buf)
            else:
                result.append(chunk)

        # 合并小于min_size的碎片，避免过碎
        final_result = []
        buffer = ""
        for chunk in result:
            if len(buffer) + len(chunk) < self.min_size:
                buffer += chunk
            else:
                if buffer:
                    final_result.append(buffer)
                    buffer = chunk
                final_result.append(chunk)
        if buffer:
            final_result.append(buffer)

        # 过滤空块
        return [c for c in final_result if c.strip()]

    def split(self, text, source=""):
        # 1. 基础递归切分，和递归切分的基础块大小一致
        base_chunks = self.base_recursive.split(text)
        # 2. 语义合并，先把所有同语义的块合起来（这次没有长度限制！）
        semantic_merged = self._local_semantic_merge(base_chunks)
        # 3. 长度归一化，严格按句子边界切分，保证完整句
        final_chunks = self._strict_length_normalize(semantic_merged)
        # 4. 生成Chunk对象
        return [Chunk(content=f, metadata={"method": "hybrid_maxmin"}, chunk_id=f"hybrid_{i}") for i, f in enumerate(final_chunks)]

# ===================== 80条 标准答案QA =====================
MEDICAL_QA = [
    {"question": "内分泌系统主要由哪些部分组成？", "answer": "内分泌腺（垂体、甲状腺、甲状旁腺、胰岛、肾上腺、性腺等）和分布在脑（下丘脑）、脂肪、心血管、呼吸道、消化道、泌尿生殖系统等的内分泌组织和细胞"},
    {"question": "激素根据化学结构分为哪四类？", "answer": "肽类和蛋白质激素、胺类和氨基酸衍生物激素、类固醇激素、脂肪酸衍生物"},
    {"question": "肽类和蛋白质激素的组成基础是什么？", "answer": "均由氨基酸残基组成分子的一级结构"},
    {"question": "胺类激素主要包括哪些？", "answer": "肾上腺素、去甲肾上腺素、血清素（5‑羟色胺）和褪黑素"},
    {"question": "类固醇激素的合成前体物质是什么？", "answer": "胆固醇"},
    {"question": "脂肪酸衍生物激素主要包括哪几类？", "answer": "前列腺素类、血栓素类、白三烯类"},
    {"question": "肽类激素的合成释放主要方式是什么？", "answer": "经基因转录、翻译成为前体，经裂解或加工形成活性物质，储存于囊泡，接收信号后通过胞吐作用排出"},
    {"question": "激素的分泌方式主要有哪六种？", "answer": "内分泌、旁分泌、自分泌、胞内分泌、神经内分泌、腔分泌"},
    {"question": "激素分泌的两大特点是什么？", "answer": "节律性、脉冲性"},
    {"question": "血液循环中的激素有哪两种存在状态？", "answer": "游离状态和结合状态"},
    {"question": "哪种激素是合成后储存可满足2个月生理需要的特例？", "answer": "甲状腺激素"},
    {"question": "激素作用的受体分为哪两大类？", "answer": "胞内（核或胞质）受体、膜受体"},
    {"question": "激素最主要的调节方式是什么？", "answer": "反馈调节"},
    {"question": "下丘脑‑垂体‑靶腺轴的反馈调节包括哪两种？", "answer": "负反馈调节、正反馈调节"},
    {"question": "膜受体一般分为哪四类？", "answer": "G蛋白耦联受体、含激酶活性受体、激酶交联受体、配体门控离子通道受体"},
    {"question": "代谢包括哪两个过程？", "answer": "合成代谢、分解代谢"},
    {"question": "人体七大营养素是什么？", "answer": "碳水化合物、脂肪、蛋白质、维生素、矿物质、膳食纤维和水"},
    {"question": "宏量营养素包括哪三类？", "answer": "碳水化合物、蛋白质、脂肪"},
    {"question": "维生素分为哪两大类？", "answer": "脂溶性维生素（A、D、E、K）、水溶性维生素（B族、C）"},
    {"question": "人体必需微量元素有多少种？", "answer": "14种"},
    {"question": "每日总能量消耗由哪两部分组成？", "answer": "基础代谢能量消耗、体力活动能量消耗"},
    {"question": "基础代谢能量消耗占每日总能量消耗的比例是多少？", "answer": "50%～70%"},
    {"question": "什么是必需氨基酸？", "answer": "自身不能合成，必须从体外补给的氨基酸"},
    {"question": "矿物质分为哪两类？", "answer": "常量元素、微量元素"},
    {"question": "膳食纤维的主要作用是什么？", "answer": "促进消化道蠕动、防止便秘、排出有害物质、预防肠道肿瘤、控制体重、降低血糖和血脂"},
    {"question": "内分泌代谢病的诊断包括哪三个方面？", "answer": "功能诊断、定位诊断、病因诊断"},
    {"question": "激素相关生化检测是反映激素水平异常的什么证据？", "answer": "间接证据"},
    {"question": "激素分泌的动态试验分为哪两类？", "answer": "兴奋试验、抑制试验"},
    {"question": "内分泌疾病定位诊断的首选影像学检查是什么？", "answer": "垂体病变首选MRI，甲状腺/甲状旁腺首选超声"},
    {"question": "库欣综合征的主要实验室诊断指标是什么？", "answer": "24小时尿游离皮质醇"},
    {"question": "嗜铬细胞瘤的定性诊断指标是什么？", "answer": "血甲氧基肾上腺素（MN）和甲氧基去甲肾上腺素（NMN）"},
    {"question": "内分泌疾病病因诊断的免疫学检查常用抗体有哪些？", "answer": "GADA、TRAb、TPOAb、TgAb"},
    {"question": "ACTH兴奋试验用于检查什么？", "answer": "肾上腺皮质产生皮质醇的储备功能"},
    {"question": "小剂量地塞米松抑制试验用于检测什么？", "answer": "皮质醇是否过度分泌"},
    {"question": "甲状腺结节定性诊断的首选方法是什么？", "answer": "甲状腺细针穿刺活检（FNA）"},
    {"question": "内分泌腺功能亢进的治疗方法有哪些？", "answer": "药物治疗、手术治疗、核素治疗、放射治疗、介入/消融治疗、分子靶向治疗、免疫治疗"},
    {"question": "内分泌腺功能减退最主要的治疗方法是什么？", "answer": "激素替代治疗"},
    {"question": "甲亢的首选抗甲状腺药物是什么？", "answer": "甲巯咪唑（MMI）、丙硫氧嘧啶（PTU）"},
    {"question": "催乳素瘤的首选药物是什么？", "answer": "多巴胺受体激动剂（溴隐亭、卡麦角林）"},
    {"question": "肢端肥大症的首选药物是什么？", "answer": "生长抑素类似物"},
    {"question": "中枢性尿崩症的首选治疗药物是什么？", "answer": "去氨加压素（DDAVP）"},
    {"question": "SIADH的首选对症治疗是什么？", "answer": "限制水摄入"},
    {"question": "甲状腺功能减退症的首选替代药物是什么？", "answer": "左甲状腺素（L‑T4）"},
    {"question": "原发性甲旁亢的首选治疗方法是什么？", "answer": "手术治疗"},
    {"question": "高钙危象的首要处理措施是什么？", "answer": "大量补液（生理盐水）扩容"},
    {"question": "下丘脑是人体的什么中枢？", "answer": "神经内分泌中枢"},
    {"question": "下丘脑分泌的抗利尿激素（ADH）由什么核团合成？", "answer": "视上核、室旁核"},
    {"question": "下丘脑的饱食中枢位于哪里？", "answer": "腹内侧核"},
    {"question": "下丘脑的摄食中枢位于哪里？", "answer": "下丘脑外侧区"},
    {"question": "下丘脑综合征的核心病因是什么？", "answer": "各种病理因素导致下丘脑神经核团和神经纤维损伤"},
    {"question": "垂体瘤最常见的三种类型是什么？", "answer": "催乳素（PRL）瘤、生长激素（GH）瘤、无功能垂体瘤"},
    {"question": "垂体瘤按大小分为哪两类？", "answer": "微腺瘤（直径＜10mm）、大腺瘤（直径≥10mm）"},
    {"question": "催乳素瘤的典型女性临床表现是什么？", "answer": "闭经、溢乳"},
    {"question": "肢端肥大症的病因是什么？", "answer": "垂体生长激素（GH）肿瘤过度分泌GH"},
    {"question": "肢端肥大症诊断的金标准是什么？", "answer": "口服葡萄糖耐量试验后GH不能被抑制至＜1μg/L"},
    {"question": "腺垂体功能减退症最常见的病因是什么？", "answer": "垂体及其附近肿瘤压迫"},
    {"question": "希恩综合征的病因是什么？", "answer": "围生期大出血导致垂体前叶缺血坏死"},
    {"question": "腺垂体功能减退症最先缺乏的激素是什么？", "answer": "生长激素（GH）、促性腺激素（LH/FSH）"},
    {"question": "生长激素缺乏性矮小症的核心诊断指标是什么？", "answer": "GH激发试验后峰值低于5μg/L"},
    {"question": "垂体卒中的核心表现是什么？", "answer": "突发剧烈头痛、恶心呕吐、视力急剧减退、意识障碍"},
    {"question": "中枢性尿崩症的病因是什么？", "answer": "精氨酸加压素（AVP/ADH）完全或部分缺乏"},
    {"question": "尿崩症的典型临床表现是什么？", "answer": "多尿、烦渴、多饮、低比重尿"},
    {"question": "禁水‑加压素试验中，中枢性尿崩症注射加压素后尿渗透压变化是什么？", "answer": "较注射前增加10%以上"},
    {"question": "SIADH的核心病理生理改变是什么？", "answer": "AVP分泌异常增多导致水潴留、低钠血症"},
    {"question": "SIADH与脑盐耗综合征的核心区别是什么？", "answer": "SIADH血容量正常，脑盐耗综合征血容量降低"},
    {"question": "甲亢最常见的病因是什么？", "answer": "毒性弥漫性甲状腺肿（Graves病）"},
    {"question": "Graves病的致病性抗体是什么？", "answer": "甲状腺刺激性抗体（TSAb）"},
    {"question": "甲亢危象的核心治疗药物是什么？", "answer": "丙硫氧嘧啶（PTU）、碘剂、普萘洛尔、糖皮质激素"},
    {"question": "Graves眼病的核心评估指标是什么？", "answer": "临床活动性评分（CAS）"},
    {"question": "原发性甲减的病因是什么？", "answer": "甲状腺本身病变，占全部甲减的99%以上"},
    {"question": "甲减的典型黏液性水肿是什么类型？", "answer": "非凹陷性水肿"},
    {"question": "自身免疫性甲状腺炎最常见的类型是什么？", "answer": "桥本甲状腺炎（HT）"},
    {"question": "亚急性甲状腺炎的特征性检查结果是什么？", "answer": "血清甲状腺激素升高、131I摄取率降低（分离现象）"},
    {"question": "非毒性甲状腺肿的核心病因是什么？", "answer": "碘缺乏"},
    {"question": "甲状腺癌最常见的病理类型是什么？", "answer": "甲状腺乳头状癌（PTC）"},
    {"question": "原发性甲旁亢的典型生化改变是什么？", "answer": "高钙血症、低磷血症、高PTH血症"},
    {"question": "原发性甲旁亢最常见的病理类型是什么？", "answer": "甲状旁腺腺瘤（约占80%～85%）"},
    {"question": "原发性甲旁亢最常见的并发症是什么？", "answer": "泌尿系统结石"},
    {"question": "甲状旁腺功能减退症的典型生化改变是什么？", "answer": "低钙血症、高磷血症、PTH降低"},
    {"question": "甲旁减的典型神经肌肉体征是什么？", "answer": "面神经叩击征（Chvostek征）阳性、束臂加压试验（Trousseau征）阳性"}
]

# ===================== 自适应命中逻辑评估器 =====================
class Evaluator:
    def __init__(self, model):
        self.model = model
        self.stop_words = {"的", "是", "为", "和", "与", "及", "或", "在", "对", "由", "什么", "哪些", "主要", "核心", "包括", "分为"}

    def build_index(self, chunks):
        texts = [c.content for c in chunks]
        encode_texts = [f"为这个句子生成表示以用于检索相关文章：{t}" for t in texts]
        emb = self.model.encode(
            encode_texts,
            normalize_embeddings=True,
            batch_size=128,
            show_progress_bar=False
        ).astype(np.float32)
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        return index, texts

    def _extract_keywords(self, text):
        terms = [t.strip() for t in re.split(r'[、，。；（）()\s\-\u2011]', text) if len(t.strip()) > 1]
        return [t for t in terms if t not in self.stop_words]

    def eval(self, chunks, qa, top_k=Config.TOP_K):
        if not chunks: return {"HitRate@3": 0, "MRR": 0}
        index, texts = self.build_index(chunks)
        hit, mrr = 0, 0.0

        questions = [f"为这个句子生成表示以用于检索相关文章：{q['question']}" for q in qa]
        q_embs = self.model.encode(
            questions,
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=False
        ).astype(np.float32)

        for q_idx, q in enumerate(qa):
            ans = q["answer"]
            q_emb = q_embs[q_idx:q_idx+1]
            _, idx = index.search(q_emb, top_k)
            ok, rank = False, -1

            key_terms = self._extract_keywords(ans)
            core_terms = [t for t in key_terms if len(t) >= 2]
            if not core_terms:
                continue

            # 自适应规则：短答案≥1个，长答案≥2个
            required_matches = min(2, len(core_terms))

            for i, index_pos in enumerate(idx[0]):
                if index_pos >= len(texts):
                    continue
                chunk_content = texts[index_pos]
                match_count = sum(1 for term in core_terms if term in chunk_content)

                if match_count >= required_matches:
                    ok = True
                    rank = i + 1
                    break

            if ok:
                hit += 1
                mrr += 1 / rank if rank > 0 else 0

        return {
            "HitRate@3": round(hit / len(qa), 4),
            "MRR": round(mrr / len(qa), 4)
        }

    def structure(self, chunks):
        lengths = [len(c.content) for c in chunks]
        end_flags = ('。', '！', '？', '”')
        full_sent = sum(1 for c in chunks if c.content.strip().endswith(end_flags))
        full_rate = round(full_sent / len(chunks), 3) if chunks else 0

        coherence = 0.0
        if len(chunks) >= 2:
            texts = [c.content for c in chunks]
            emb = self.model.encode(texts, batch_size=128, show_progress_bar=False)
            sims = [cosine_similarity([emb[i]], [emb[i+1]])[0][0] for i in range(len(chunks)-1)]
            coherence = round(np.mean(sims), 3)

        return {
            "块数量": len(chunks),
            "平均长度": round(np.mean(lengths), 1),
            "标准差": round(np.std(lengths), 1),
            "碎片(<100)": sum(1 for l in lengths if l < 100),
            "超长块(>800)": sum(1 for l in lengths if l > 800),
            "完整句比例": full_rate,
            "语义连贯性": coherence
        }

# ===================== 主程序 =====================
if __name__ == "__main__":
    try:
        with open(Config.FILE_PATH, "r", encoding="utf-8") as f:
            text = f.read()
        print("✅ 文本加载成功，总长度：", len(text))
    except Exception as e:
        print(f"❌ 文件读取失败：{e}")
        exit()

    print(f"🔽 加载模型：BAAI/bge-base-zh-v1.5 | 设备：{DEVICE} | 半精度：{USE_FP16}")
    model = SentenceTransformer(
        "BAAI/bge-base-zh-v1.5",
        device=DEVICE,
        trust_remote_code=True
    )
    model.eval()
    torch.set_grad_enabled(False)
    if USE_FP16:
        model.half()

    chunkers = {
        "固定切分": FixedSizeChunker(),
        "递归切分": RecursiveChunker(),
        "传统语义切分": OriginalSemanticChunker(model),
        "🔥自研MaxMin混合切分": RecursiveMaxMinHybridChunker(model)
    }

    evaluator = Evaluator(model)
    final_result = []

    print("\n🚀 开始量化评估（80条QA自适应命中规则）...\n")
    for name, chker in chunkers.items():
        print(f"正在评估：{name}")
        chunks = chker.split(text)
        ret = evaluator.eval(chunks, MEDICAL_QA)
        stru = evaluator.structure(chunks)
        final_result.append({"方法": name, **stru, **ret})
        print(f"  块数量：{stru['块数量']} | 完整句比例：{stru['完整句比例']} | HitRate@3：{ret['HitRate@3']} | MRR：{ret['MRR']}\n")

    # 打印结果
    df = pd.DataFrame(final_result)
    from tabulate import tabulate

    df_print = df.copy()
    format_rules = {
        "HitRate@3": "{:.4f}",
        "MRR": "{:.4f}",
        "平均长度": "{:.1f}",
        "标准差": "{:.1f}",
        "完整句比例": "{:.3f}",
        "语义连贯性": "{:.3f}"
    }
    for col, fmt in format_rules.items():
        if col in df_print.columns:
            df_print[col] = df_print[col].apply(lambda x: fmt.format(x))

    print("\n" + "="*150)
    print("📊 文本切分方法评估最终结果【自适应命中+自研修复版】")
    print("="*150)
    print(tabulate(
        df_print,
        headers="keys",
        tablefmt="pipe",
        stralign="center",
        numalign="center",
        showindex=False
    ))

    df.to_excel("文本切分实验结果_最终版.xlsx", index=False)
    print("\n✅ 结果已保存：文本切分实验结果_最终版.xlsx")

    # 绘图
    import matplotlib.pyplot as plt
    import numpy as np

    method_names = [res["方法"] for res in final_result]
    hit_rates    = [res["HitRate@3"] for res in final_result]
    mrrs         = [res["MRR"] for res in final_result]
    avg_lengths  = [res["平均长度"] for res in final_result]
    std_lengths  = [res["标准差"] for res in final_result]

    x = np.arange(len(method_names))
    bar_width = 0.6

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 200

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    best_hit = hit_rates.index(max(hit_rates))
    best_mrr = mrrs.index(max(mrrs))
    colors_hit = ['#D0D0D0'] * len(method_names)
    colors_mrr = ['#D0D0D0'] * len(method_names)
    colors_hit[best_hit] = '#2E8B57'
    colors_mrr[best_mrr] = '#4169E1'

    ax1.bar(x, hit_rates, bar_width, color=colors_hit)
    ax1.set_title('HitRate@3 对比', fontweight='bold', fontsize=12)
    ax1.set_ylabel('检索召回率')
    ax1.set_xticks(x)
    ax1.set_xticklabels(method_names, rotation=15, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    ax1.spines[['top','right']].set_visible(False)

    ax2.bar(x, mrrs, bar_width, color=colors_mrr)
    ax2.set_title('MRR 对比', fontweight='bold', fontsize=12)
    ax2.set_ylabel('排序精度')
    ax2.set_xticks(x)
    ax2.set_xticklabels(method_names, rotation=15, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    ax2.spines[['top','right']].set_visible(False)

    ax3.errorbar(x, avg_lengths, yerr=std_lengths, fmt='o-', color='#DC143C', capsize=4, linewidth=2)
    ax3.set_title('平均长度与均匀性', fontweight='bold', fontsize=12)
    ax3.set_ylabel('字符数')
    ax3.set_xticks(x)
    ax3.set_xticklabels(method_names, rotation=15, ha='right')
    ax3.grid(axis='y', alpha=0.3)
    ax3.spines[['top','right']].set_visible(False)

    plt.tight_layout()
    plt.savefig("实验结果图_最终版.svg", bbox_inches='tight')
    plt.savefig("实验结果图_最终版.png", bbox_inches='tight')
    plt.show()
