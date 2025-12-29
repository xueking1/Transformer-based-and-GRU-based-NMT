# 导入SentencePiece库，该库是谷歌开源的无监督文本分词工具，常用于NLP任务的子词建模
import sentencepiece as spm


def train(input_file, vocab_size, model_name, model_type, character_coverage):
    """
    训练SentencePiece分词模型的核心函数
    参考文档：https://github.com/google/sentencepiece/blob/master/doc/options.md
    :param input_file: 输入语料文件路径，要求每行一个句子，无需提前分词/归一化（工具会自动用Unicode NFKC归一化）
                    支持传入逗号分隔的多个文件路径（如"corpus1.en,corpus2.en"）
    :param vocab_size: 词汇表大小，常用值为8000/16000/32000，越大表示子词粒度越细
    :param model_name: 输出模型的前缀名，训练后会生成 <model_name>.model（模型文件）和 <model_name>.vocab（词汇表文件）
    :param model_type: 模型类型，可选值：
                    - unigram（默认）：基于概率的无igram模型，适合NMT等任务
                    - bpe：字节对编码，适合大多数NLP场景
                    - char：字符级模型，粒度最细
                    - word：单词级模型（需提前对输入文本分词）
    :param character_coverage: 模型覆盖的字符比例，核心参数：
                            - 中文/日文等字符集丰富的语言：0.9995（覆盖99.95%的字符）
                            - 英文等字符集小的语言：1.0（覆盖全部字符）
    """
    # 构造SentencePiece训练命令的参数模板
    # 额外指定特殊标记的ID：pad(填充)=0、unk(未知词)=1、bos(句首)=2、eos(句尾)=3
    input_argument = '--input=%s --model_prefix=%s --vocab_size=%s --model_type=%s --character_coverage=%s ' \
                    '--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 '
    # 将传入的参数填充到命令模板中，生成完整的训练命令
    cmd = input_argument % (input_file, model_name, vocab_size, model_type, character_coverage)
    # 执行SentencePiece模型训练
    spm.SentencePieceTrainer.Train(cmd)


def run():
    """
    主训练函数：分别训练英文和中文的BPE分词模型
    """
    # ---------------------- 英文模型训练配置 ----------------------
    en_input = '../data/corpus.en'          # 英文语料文件路径（对应之前生成的corpus.en）
    en_vocab_size = 32000                   # 英文词汇表大小设为32000
    en_model_name = 'eng'                   # 英文模型前缀名，训练后生成eng.model和eng.vocab
    en_model_type = 'bpe'                   # 采用BPE（字节对编码）模型
    en_character_coverage = 1               # 英文字符集小，覆盖100%字符
    # 调用训练函数训练英文模型
    train(en_input, en_vocab_size, en_model_name, en_model_type, en_character_coverage)

    # ---------------------- 中文模型训练配置 ----------------------
    ch_input = '../data/corpus.ch'          # 中文语料文件路径（对应之前生成的corpus.ch）
    ch_vocab_size = 32000                   # 中文词汇表大小设为32000
    ch_model_name = 'chn'                   # 中文模型前缀名，训练后生成chn.model和chn.vocab
    ch_model_type = 'bpe'                   # 采用BPE（字节对编码）模型
    ch_character_coverage = 0.9995          # 中文字符集丰富，覆盖99.95%字符即可
    # 调用训练函数训练中文模型
    train(ch_input, ch_vocab_size, ch_model_name, ch_model_type, ch_character_coverage)


def test():
    """
    测试函数：验证训练好的中文模型的编码/解码功能
    """
    # 初始化SentencePiece处理器（用于加载模型和执行分词/还原）
    sp = spm.SentencePieceProcessor()
    # 测试文本：中文句子
    text = "美国总统特朗普今日抵达夏威夷。"

    # 加载训练好的中文模型文件（chn.model）
    sp.Load("./32000/chn.model")
    # sp.Load("./ChineseNMT/tokenizer/chn.model")
    # 测试1：将文本编码为子词片段（可视化分词结果）
    print("文本分词后的子词片段：", sp.EncodeAsPieces(text))
    # 测试2：将文本编码为子词对应的ID（模型输入常用格式）
    print("文本分词后的ID序列：", sp.EncodeAsIds(text))
    # 测试3：将ID序列解码回文本（验证解码功能）
    a = [24538, 7546, 6726, 30241, 29334, 31380, 28865]
    print("ID序列解码后的文本：", sp.decode_ids(a))


# 程序入口：执行训练流程
if __name__ == "__main__":
    # run()
    test()