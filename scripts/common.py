# -*- coding: utf-8 -*-
"""
通用工具模块（common.py）
========================
本文件把"所有脚本都要用到"的小工具集中在一起，避免每个脚本重复写一遍：
  1) 路径常量：原始数据、处理后数据的存放位置；
  2) 字段读取：兼容训练集里存在的【两套不一致的 schema】；
  3) 语言识别：粗略判断一段文本主要是哪种语言；
  4) 词数统计：用于判断"短文本（<100 词）"子集。

为什么要专门处理 schema？
  我们实测发现 data.jsonl 其实是多个子数据集拼起来的，字段名并不统一：
    - A 型（摘要型）：id / url / title / text / machine_abstract / machine_text
                      / prompt / model / source / score / probas / logits
    - B 型（问答/文章型）：prompt / human_text / machine_text / model / source
                           / source_ID（有的写成 source_id）
  如果代码只认其中一套，就会漏掉一大半数据。所以这里统一封装成
  get_human_text / get_machine_text / get_domain / get_generator。
"""

import os
import re

# ---------------------------------------------------------------------------
# 1) 路径常量
# ---------------------------------------------------------------------------
# 本文件位于 <项目根目录>/scripts/ 下，所以向上一级就是项目根目录。
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 原始训练数据（组委会提供的 data.jsonl）
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "data.jsonl")
# 预处理产物统一放在 data/processed/ 下
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


# ---------------------------------------------------------------------------
# 2) 字段读取（兼容两套 schema）
# ---------------------------------------------------------------------------

def schema_of(rec):
    """判断一条记录属于哪套 schema，便于统计与调试。
    返回 'abstract'（A 型，含 machine_abstract）、'human_text'（B 型）或 'other'。
    """
    if isinstance(rec.get("machine_abstract"), str):
        return "abstract"
    if isinstance(rec.get("human_text"), str):
        return "human_text"
    return "other"


def get_human_text(rec):
    """取"人类撰写"的正文。
    A 型字段名是 text；B 型字段名是 human_text。取不到就返回空字符串。
    """
    for key in ("text", "human_text"):
        val = rec.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""


def get_machine_text(rec):
    """取"机器生成"的正文（尽量去掉提示词前缀）。
    A 型：machine_abstract 本身就是"正文（不含提示词前缀）"，直接用。
    B 型：只有 machine_text，可能带提示词前缀，需尝试剥离。
    """
    abs_txt = rec.get("machine_abstract")
    if isinstance(abs_txt, str) and abs_txt.strip():
        return abs_txt
    mt = rec.get("machine_text")
    if isinstance(mt, str) and mt.strip():
        return _strip_prompt_prefix(mt, rec)
    return ""


def _strip_prompt_prefix(machine_text, rec):
    """尝试把 machine_text 开头的提示词模板去掉，只保留正文。
    做法：把 prompt 里的 {title} 用真实 title 替换，得到"具体前缀"，
         若 machine_text 确实以它开头就切掉。切不掉就原样返回。
    """
    prompt = rec.get("prompt")
    title = rec.get("title", "")
    if not isinstance(prompt, str) or not prompt:
        return machine_text
    candidates = [prompt]
    if isinstance(title, str) and title:
        candidates.append(prompt.replace("{title}", title))
        candidates.append(prompt.replace("{title}", title).rstrip())
    for pref in candidates:
        if pref and machine_text.startswith(pref):
            # 切掉前缀，并去掉紧跟的换行/空白，得到干净的正文
            return machine_text[len(pref):].lstrip("\n \t")
    return machine_text


def get_domain(rec):
    """取数据领域/来源（如 wikipedia、arxiv、wikihow…）。取不到记作 'unknown'。"""
    v = rec.get("source")
    if isinstance(v, str) and v.strip():
        return v.strip()
    return "unknown"


def get_generator(rec):
    """取机器文本的生成器名称（如 gpt-3.5-turbo）。取不到记作 'unknown'。"""
    v = rec.get("model")
    if isinstance(v, str) and v.strip():
        return v.strip()
    return "unknown"


def get_source_id(rec):
    """取"源文章"的唯一标识。不同 schema 字段名大小写不一致，这里都兼容。
    优先用 source_ID / source_id，其次用 id。都没有则返回空字符串。
    """
    for key in ("source_ID", "source_id", "id"):
        v = rec.get(key)
        if v is not None and str(v).strip():
            return str(v)
    return ""


# ---------------------------------------------------------------------------
# 2.5) 文本归一化（跨域实验用）
# ---------------------------------------------------------------------------
# 【为什么做】跨域掉分的主要表现是 machine 被判成 human：模型在训练域里
# 把"某种排版痕迹"当成了机器信号，换到一个没见过的域，这种痕迹消失或变形，
# 判定就整体左移。把"域相关但和作者无关"的结构标记（链接、公式、引用编号、
# 代码块、HTML 标签）统一替换为占位符，可以迫使模型改学更通用的写作风格信号。
#
# 【注意】训练、验证、推理必须用同一套规则，否则输入分布不一致会直接掉分。
_RE_FENCE = re.compile(r"```[\s\S]{0,3000}?```")
_RE_LATEX_ENV = re.compile(r"\\begin\{[^}]{1,40}\}[\s\S]{0,1500}?\\end\{[^}]{1,40}\}")
_RE_LATEX_DISPLAY = re.compile(r"\$\$[\s\S]{0,800}?\$\$")
_RE_LATEX_INLINE = re.compile(r"\$[^$\n]{1,200}\$")
_RE_LATEX_CMD = re.compile(r"\\[a-zA-Z]{1,30}\*?")
_RE_CITE = re.compile(r"\[\d+(?:\s*[,;–-]\s*\d+)*\]")
_RE_URL = re.compile(r"(?:https?://|www\.)[^\s<>\"')]{1,300}")
_RE_HTML = re.compile(r"</?[a-zA-Z][^>\n]{0,80}>")
_RE_MD_HEAD = re.compile(r"(?m)^[ \t]{0,3}#{1,6}[ \t]*")
_RE_BLANK = re.compile(r"[ \t]{2,}")
_RE_NL = re.compile(r"\n{3,}")


def normalize_text(text):
    """把域相关的排版标记替换成占位符，保留正文内容与句法结构。"""
    if not text:
        return ""
    t = _RE_FENCE.sub(" [CODE] ", text)
    t = _RE_LATEX_ENV.sub(" [MATH] ", t)
    t = _RE_LATEX_DISPLAY.sub(" [MATH] ", t)
    t = _RE_LATEX_INLINE.sub(" [MATH] ", t)
    # LaTeX 命令只替换命令名本身，后面 {...} 里的正文保留（那才是写作风格信号）
    t = _RE_LATEX_CMD.sub(" [TEX] ", t)
    t = _RE_CITE.sub(" [CITE] ", t)
    t = _RE_URL.sub(" [URL] ", t)
    t = _RE_HTML.sub(" ", t)
    t = _RE_MD_HEAD.sub("", t)
    t = _RE_BLANK.sub(" ", t)
    t = _RE_NL.sub("\n\n", t)
    return t.strip()


# ---------------------------------------------------------------------------
# 3) 语言识别（粗略）
# ---------------------------------------------------------------------------
# 用 Unicode 区块统计各类字符数量，谁多就判为谁。够用即可，不做精细语种识别。
_RE_HAN = re.compile(r"[\u4e00-\u9fff]")            # 汉字
_RE_ARABIC = re.compile(r"[\u0600-\u06ff]")         # 阿拉伯字母
_RE_CYRILLIC = re.compile(r"[\u0400-\u04ff]")       # 西里尔字母
_RE_LATIN_EXT = re.compile(r"[áéíóúñüç¿¡ÁÉÍÓÚÑÜ]")  # 带重音的拉丁字母（西语等）
_RE_LATIN = re.compile(r"[A-Za-z]")


def detect_lang(text):
    """粗略判断语言，返回 'zh' / 'ar' / 'ru' / 'es' / 'en' / 'empty'。"""
    if not text or not text.strip():
        return "empty"
    counts = {
        "zh": len(_RE_HAN.findall(text)),
        "ar": len(_RE_ARABIC.findall(text)),
        "ru": len(_RE_CYRILLIC.findall(text)),
        "es": len(_RE_LATIN_EXT.findall(text)),
        "en": len(_RE_LATIN.findall(text)),
    }
    lang = max(counts, key=counts.get)
    return lang if counts[lang] > 0 else "en"


# ---------------------------------------------------------------------------
# 4) 词数统计（用于"短文本 <100 词"子集）
# ---------------------------------------------------------------------------
_RE_LATIN_WORD = re.compile(r"[A-Za-z0-9']+")
_RE_CJK = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")


def count_words(text):
    """统计"词数"。
    英文等空格分词语言：按单词切分计数；
    中文/日文/韩文：按字符计数（一个字近似当作一个"词"）。
    最终 = 拉丁词数 + CJK 字符数。这是对赛题"<100 词"的近似口径，需在报告中说明。
    """
    if not text:
        return 0
    return len(_RE_LATIN_WORD.findall(text)) + len(_RE_CJK.findall(text))
