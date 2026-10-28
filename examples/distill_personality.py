import json
import os

from bioelectric.llm_annotate import LLMAnnotator
from bioelectric.learned_personality import train_emotion_mlp
from bioelectric.personality_eval import HOLD_OUT_TESTS, evaluate_quadrant_accuracy


# 待标注语句：覆盖四象限 + 成语/近义/口语/抽象表达
CORPUS = [
    "今天太开心了", "我真的很愤怒", "我很委屈", "现在很放松", "我很焦虑",
    "欢呼雀跃", "心平气和", "怒不可遏", "郁郁寡欢", "怡然自得",
    "太令人惊喜了", "心里很踏实", "气得浑身发抖", "心里空落落的",
    "一切都好起来了", "感觉挺安稳的", "简直要气疯了", "整个人没精神",
    "真让人振奋", "这真是天大的好事", "吓死我了", "感觉被掏空了",
    "喜出望外", "安之若素", "暴跳如雷", "黯然神伤",
    "心情舒畅", "悠然自得", "怒发冲冠", "垂头丧气",
    # ==== 扩充：更多成语/口语/抽象/中性表达 ====
    "喜气洋洋", "春风得意", "笑逐颜开", "乐不思蜀", "神采奕奕",
    "心如止水", "宠辱不惊", "泰然自若", "波澜不惊", "气定神闲",
    "火冒三丈", "怒目而视", "雷霆大怒", "气冲牛斗", "怒形于色",
    "忧心忡忡", "忐忑不安", "惶恐不安", "寝食难安", "忧心如焚",
    "闷闷不乐", "无精打采", "垂头丧气", "灰心丧气", "唉声叹气",
    "好开心呀", "真高兴", "太激动了", "心情美滋滋", "乐坏了",
    "烦死了", "气死我了", "怕得要命", "急死了", "担心死了",
    "有点失落", "不太舒服", "挺郁闷的", "好孤单", "心好累",
    "感觉还不错", "还行吧", "没什么感觉", "一般般", "凑合",
    "惊喜连连", "兴奋不已", "欢欣鼓舞", "雀跃欢呼", "喜上眉梢",
    "心如刀割", "悲痛欲绝", "万箭穿心", "肝肠寸断", "痛彻心扉",
]


def main():
    cache_path = "data/distill_annotations.json"
    annotator = LLMAnnotator(model="qwen3.7-plus")

    # 缓存：已标注过的语句不重复调 LLM
    cache = {}
    if os.path.exists(cache_path):
        cache = json.load(open(cache_path, encoding="utf-8"))

    labels = []
    new_texts = []
    for t in CORPUS:
        if t in cache:
            labels.append(tuple(cache[t]))
        else:
            new_texts.append(t)
            labels.append(None)

    if new_texts:
        print(f"用 LLM 标注 {len(new_texts)} 条新语句...")
        for i, t in enumerate(new_texts):
            lab = annotator.annotate(t)
            cache[t] = list(lab) if lab else None
            idx = CORPUS.index(t)
            labels[idx] = lab
        json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # 过滤失败的标注
    texts, valid_labels = [], []
    for t, l in zip(CORPUS, labels):
        if l is not None:
            texts.append(t)
            valid_labels.append(l)
    print(f"成功标注 {len(texts)} 条")

    print("训练 MLP 蒸馏 LLM 的情绪理解...")
    model = train_emotion_mlp(texts, valid_labels, epochs=500, hidden=32, lr=0.05)

    acc = evaluate_quadrant_accuracy(model)
    print(f"四象限准确率（未见面测试）: {acc:.2f}")
    print()
    print("== 未见面语句的预测 ==")
    for text, v, a in HOLD_OUT_TESTS:
        pv, pa = model.predict_text(text)
        print(f"  {text:<14} 标签=({v:+.1f},{a:+.1f})  预测=({pv:+.2f},{pa:+.2f})")


if __name__ == "__main__":
    main()
