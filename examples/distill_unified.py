import json
import os

from bioelectric.llm_annotate import LLMAnnotator
from bioelectric.unified_personality import train_unified_mlp, DIMENSIONS


# 语料：覆盖情绪四象限 + 各类自我陈述
CORPUS = [
    "今天太开心了", "我真的很愤怒", "我很委屈", "现在很放松", "我很焦虑",
    "欢呼雀跃", "心平气和", "怒不可遏", "郁郁寡欢", "怡然自得",
    "我清楚地知道自己是谁", "我做事很有主见", "我总是犹豫不决",
    "我很内向，喜欢独处", "我很外向，喜欢社交", "我做事情很认真负责",
    "我对新事物很感兴趣", "我情绪很容易波动", "我很随和好相处",
    "我常常反思自己", "我感到迷茫", "我目标很明确",
    "喜出望外", "安之若素", "暴跳如雷", "黯然神伤",
    "心情舒畅", "悠然自得", "怒发冲冠", "垂头丧气",
]


def main():
    cache_path = "data/distill_unified_annotations.json"
    annotator = LLMAnnotator(model="qwen3.7-plus")

    cache = {}
    if os.path.exists(cache_path):
        cache = json.load(open(cache_path, encoding="utf-8"))

    labels = [None] * len(CORPUS)
    new_texts = []
    for i, t in enumerate(CORPUS):
        if t in cache:
            labels[i] = tuple(cache[t])
        else:
            new_texts.append((i, t))

    if new_texts:
        print(f"LLM 标注 {len(new_texts)} 条新语句（9 维）...")
        for i, t in new_texts:
            lab = annotator.annotate_unified(t)
            cache[t] = list(lab) if lab else None
            labels[i] = lab
        json.dump(cache, open(cache_path, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)

    # 过滤失败标注
    texts, valid_labels = [], []
    for t, l in zip(CORPUS, labels):
        if l is not None:
            texts.append(t)
            valid_labels.append(list(l))
    print(f"成功标注 {len(texts)} 条")

    print("训练 9 维统一人格模型...")
    model = train_unified_mlp(texts, valid_labels, epochs=500, hidden=32, lr=0.05)
    model.save("data/unified_personality_model.npz")
    print("模型已保存到 data/unified_personality_model.npz")

    # 演示
    print("\n== 9 维人格向量演示 ==")
    for t in ["我清楚地知道自己是谁", "我真的很愤怒", "现在很放松", "我做事情很认真负责"]:
        out = model.predict_text(t)
        print(f"  {t}")
        print(f"    情绪={out[0]:+.2f},{out[1]:+.2f} 性格={[round(x,2) for x in out[2:7]]} 意识={[round(x,2) for x in out[7:9]]}")


if __name__ == "__main__":
    main()
