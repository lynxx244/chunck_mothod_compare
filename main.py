
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
from tabulate import tabulate
import matplotlib.pyplot as plt

# ===================== 数据结构定义 =====================
@dataclass
class Chunk:
    content: str
    metadata: Dict = field(default_factory=dict)
    chunk_id: str = ""

# ===================== 1. 固定长度切分 =====================
class FixedSizeChunker:
    def __init__(self, chunk_size=500, overlap=50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text):
        chunks = []
        start, idx = 0, 0
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(Chunk(content=chunk_text, metadata={"method": "fixed"}, chunk_id=f"fixed_{idx}"))
                idx += 1
            start = end - self.overlap
        return chunks

# ===================== 2. 递归结构切分 =====================
class RecursiveChunker:
    def __init__(self, chunk_size=500, overlap=50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = ["\n\n", "\n", "。", "！", "？", "；", " "]

    def split(self, text):
        chunks = self._recurse(text, 0)
        return [Chunk(content=c.strip(), metadata={"method": "recursive"}, chunk_id=f"recur_{i}") 
                for i, c in enumerate(chunks) if c.strip()]

    def _recurse(self, text, sep_idx):
        if len(text) <= self.chunk_size or sep_idx >= len(self.separators):
            return [text]
        sep = self.separators[sep_idx]
        splits = text.split(sep)
        chunks, current = [], ""
        for part in splits:
            part += sep
            if len(current) + len(part) <= self.chunk_size:
                current += part
            else:
                if current: chunks.append(current.strip())
                current = part
                if len(current) > self.chunk_size:
                    chunks.extend(self._recurse(current, sep_idx+1))
                    current = ""
        if current.strip(): chunks.append(current.strip())
        return chunks

# ===================== 3. 传统语义切分 =====================
class OriginalSemanticChunker:
    def __init__(self, model, threshold=0.55, max_chunk=500, min_chunk=100):
        self.model = model
        self.threshold = threshold
        self.max_chunk = max_chunk
        self.min_chunk = min_chunk

    def _split_sentences(self, text):
        return [s.strip() for s in re.split(r'(?<=[。！？；\n])', text) if s.strip()]

    def split(self, text):
        sentences = self._split_sentences(text)
        if not sentences: return []
        emb = self.model.encode(sentences)
        chunks, current, length = [], [sentences[0]], len(sentences[0])
        for i in range(1, len(sentences)):
            sim = cosine_similarity([emb[i-1]], [emb[i]])[0][0]
            length += len(sentences[i])
            if sim < self.threshold or length > self.max_chunk:
                chunks.append("".join(current))
                current, length = [sentences[i]], len(sentences[i])
            else:
                current.append(sentences[i])
        chunks.append("".join(current))
        merged, buf = [], ""
        for c in chunks:
            if len(buf) + len(c) < self.min_chunk: buf += c
            else:
                if buf: merged.append(buf); buf = c
                merged.append(c)
        if buf: merged.append(buf)
        return [Chunk(content=m, metadata={"method": "semantic"}, chunk_id=f"sem_{i}") for i, m in enumerate(merged)]

# ===================== 4. 自研MaxMin混合切分（最优） =====================
class RecursiveMaxMinHybridChunker:
    def __init__(self, model, min_chunk=100, max_chunk=850, threshold=0.4):
        self.model = model
        self.min_size = min_chunk
        self.max_size = max_chunk
        self.threshold = threshold
        self.base_recursive = RecursiveChunker(chunk_size=400)

    def split(self, text):
        base_chunks = self.base_recursive.split(text)
        merged = self._semantic_merge(base_chunks)
        final = self._length_normalize(merged)
        return [Chunk(content=f, metadata={"method": "hybrid"}, chunk_id=f"hybrid_{i}") for i, f in enumerate(final)]

    def _semantic_merge(self, chunks):
        if len(chunks) <= 1: return [c.content for c in chunks]
        texts = [c.content for c in chunks]
        emb = self.model.encode(texts, batch_size=32)
        res, cur, cur_emb, cur_len = [], [chunks[0]], emb[0], len(texts[0])
        for i in range(1, len(chunks)):
            c, l, e = chunks[i], len(texts[i]), emb[i]
            if cur_len + l > self.max_size:
                res.append("".join([x.content for x in cur]))
                cur, cur_emb, cur_len = [c], e, l
                continue
            if cosine_similarity([cur_emb], [e])[0][0] >= self.threshold:
                cur.append(c)
                cur_emb = np.mean([cur_emb, e], axis=0)
                cur_len += l
            else:
                res.append("".join([x.content for x in cur]))
                cur, cur_emb, cur_len = [c], e, l
        if cur: res.append("".join([x.content for x in cur]))
        return res

    def _length_normalize(self, chunks):
        res, buf = [], ""
        for c in chunks:
            c = c.strip()
            if not c: continue
            if len(buf) + len(c) < self.min_size: buf += c
            else:
                if buf: res.append(buf); buf = c
                res.append(c if len(c) <= self.max_size else [c[i:i+self.max_size-50] for i in range(0,len(c),self.max_size-50)][0])
        if buf: res[-1] += buf if res else buf
        return [r for r in res if r.strip()]

# ===================== 评估器 =====================
class Evaluator:
    def __init__(self, model): self.model = model
    def eval(self, chunks, qa, top_k=3):
        if not chunks: return {"HitRate@3":0,"MRR":0}
        texts = [c.content for c in chunks]
        emb = self.model.encode(texts, normalize_embeddings=True).astype(np.float32)
        index = faiss.IndexFlatIP(emb.shape[1]); index.add(emb)
        hit, mrr = 0, 0.0
        for q in qa:
            q_emb = self.model.encode([q["question"]], normalize_embeddings=True).astype(np.float32)
            _, idx = index.search(q_emb, top_k)
            for i, pos in enumerate(idx[0]):
                if q["answer"] in texts[pos] or q["answer"][:18] in texts[pos]:
                    hit +=1; mrr += 1/(i+1); break
        return {"HitRate@3":round(hit/len(qa),4),"MRR":round(mrr/len(qa),4)}

    def structure(self, chunks):
        lengths = [len(c.content) for c in chunks]
        full = sum(1 for c in chunks if c.content.strip().endswith(('。','！','？','”')))
        full_rate = round(full/len(chunks),3) if chunks else 0
        coherence = 0.0
        if len(chunks)>=2:
            emb = self.model.encode([c.content for c in chunks])
            coherence = round(np.mean([cosine_similarity([emb[i]],[emb[i+1]])[0][0] for i in range(len(chunks)-1)]),3)
        return {
            "块数量":len(chunks),"平均长度":round(np.mean(lengths),1),"标准差":round(np.std(lengths),1),
            "碎片(<100)":sum(1 for l in lengths if l<100),"超长块(>800)":sum(1 for l in lengths if l>800),
            "完整句比例":full_rate,"语义连贯性":coherence
        }

# ===================== 80条医学QA,由大模型针对原文本生成 =====================
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

# ===================== 主程序 =====================
if __name__ == "__main__":
    print("="*60)
    print(" 医学文本智能切分评估工具")
    print(" 支持4种切分算法 | 9大评估指标 | 自动可视化")
    print("="*60)

    # 加载文本
    with open("内科学第10版.txt", "r", encoding="utf-8") as f:
        text = f.read()[:150000]

    # 加载模型
    model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
    evaluator = Evaluator(model)

    # 四种切分方法
    chunkers = {
        "固定切分": FixedSizeChunker(),
        "递归切分": RecursiveChunker(),
        "传统语义切分": OriginalSemanticChunker(model),
        "自研MaxMin混合切分": RecursiveMaxMinHybridChunker(model)
    }

    final_result = []
    for name, chker in chunkers.items():
        print(f"\n正在评估: {name}")
        chunks = chker.split(text)
        ret = evaluator.eval(chunks, MEDICAL_QA)
        stru = evaluator.structure(chunks)
        final_result.append({"方法": name, **stru, **ret})

    # 打印极简表格
    df = pd.DataFrame(final_result)
    print("\n📊 评估结果")
    print(tabulate(df, headers='keys', tablefmt='simple', stralign='center', numalign='center', showindex=False))

    # 保存结果
    df.to_excel("实验结果.xlsx", index=False)
    print("\n✅ 结果已保存: 实验结果.xlsx")

    # 绘图
    method_names = [r["方法"] for r in final_result]
    hit_rates = [r["HitRate@3"] for r in final_result]
    mrrs = [r["MRR"] for r in final_result]
    avg = [r["平均长度"] for r in final_result]
    std = [r["标准差"] for r in final_result]
    x = np.arange(len(method_names))

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(18,5))

    ax1.bar(x, hit_rates, color=['#ccc','#ccc','#ccc','#2E8B57'])
    ax1.set_title('HitRate@3'); ax1.set_xticks(x); ax1.set_xticklabels(method_names, rotation=15)

    ax2.bar(x, mrrs, color=['#ccc','#ccc','#ccc','#4169E1'])
    ax2.set_title('MRR'); ax2.set_xticks(x); ax2.set_xticklabels(method_names, rotation=15)

    ax3.errorbar(x, avg, yerr=std, fmt='o-', capsize=4)
    ax3.set_title('长度与标准差'); ax3.set_xticks(x); ax3.set_xticklabels(method_names, rotation=15)

    plt.tight_layout()
    plt.savefig("结果图.png", bbox_inches='tight')
    plt.show()
