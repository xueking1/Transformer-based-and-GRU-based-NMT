# encoding=utf-8
import sentencepiece as spm
import jieba
from collections import Counter
import os


def train(input_file, vocab_size, model_name, model_type, character_coverage):
    """英文BPE模型训练（原有逻辑不变，生成model_name.model + model_name.vocab）"""
    input_argument = '--input=%s --model_prefix=%s --vocab_size=%s --model_type=%s --character_coverage=%s ' \
                    '--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 '
    cmd = input_argument % (input_file, model_name, vocab_size, model_type, character_coverage)
    spm.SentencePieceTrainer.Train(cmd)


def train_jieba(input_file, vocab_size, model_name, model_type=None, character_coverage=None):
    """
    调整：保存路径与英文模型一致，文件名对齐为model_name.vocab
    （jieba无需生成.model文件，因为它不需要模型文件）
    """
    jieba.enable_paddle()  # 沿用paddle模式
    word_counter = Counter()
    seg_corpus = []

    # 1. 读取语料并分词
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            seg_words = list(jieba.cut(line, use_paddle=True))
            word_counter.update(seg_words)
            seg_corpus.append(' '.join(seg_words))

    # 2. 生成词汇表（对齐特殊标记ID：pad=0/unk=1/bos=2/eos=3）
    special_tokens = ['<pad>', '<unk>', '<bos>', '<<eos>']
    word2id = {t: idx for idx, t in enumerate(special_tokens)}
    # 按词频取前vocab_size个词，从ID=4开始
    sorted_words = sorted(word_counter.items(), key=lambda x: (-x[1], x[0]))[:vocab_size]
    for idx, (word, _) in enumerate(sorted_words, start=4):
        word2id[word] = idx

    # 3. 保存文件（与英文模型同目录，文件名对齐为model_name.vocab）
    # 3.1 保存分词后语料（可选，命名为model_name.seg_corpus）
    seg_corpus_path = f"{model_name}.seg_corpus"
    with open(seg_corpus_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(seg_corpus))

    # 3.2 保存词汇表：命名为model_name.vocab（与英文模型的.vocab格式对齐）
    vocab_path = f"{model_name}.vocab"
    with open(vocab_path, 'w', encoding='utf-8') as f:
        # 先写特殊标记（格式：词\tID\t词频）
        for t in special_tokens:
            f.write(f"{t}\t{word2id[t]}\t-1\n")
        # 再写普通词汇
        for word, count in sorted_words:
            f.write(f"{word}\t{word2id[word]}\t{count}\n")
    print(f"中文词汇表已保存为：{vocab_path}")

    return word2id


def run():
    """主函数：生成eng.model/eng.vocab（英文BPE） + chn.vocab（jieba）"""
    # 英文BPE训练（生成eng.model + eng.vocab）
    train(
        input_file='../data/corpus.en',
        vocab_size=32000,
        model_name='eng',
        model_type='bpe',
        character_coverage=1
    )

    # 中文jieba处理（生成chn.vocab）
    train_jieba(
        input_file='../data/corpus.ch',
        vocab_size=32000,
        model_name='chn'  # 文件名前缀为chn，最终生成chn.vocab
    )


if __name__ == "__main__":
    run()