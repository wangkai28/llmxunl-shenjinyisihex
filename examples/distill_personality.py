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
]


def main():
    annotator = LLMAnnotator(model="qwen3.7-plus")
    print(f"用 LLM 标注 {len(CORPUS)} 条语句...")
    labels = annotator.annotate_batch(CORPUS)

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
