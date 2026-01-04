# Transformer-based and GRU-based Model

基于GRU和transformer的英译中翻译模型。

## Data

采用100k和10k文本数据集。

## Data Process

### 分词

- 工具：[sentencepiece](https://github.com/google/sentencepiece)
- 预处理：`./data/get_corpus.py`抽取train、dev和test中双语语料，分别保存到`corpus.en`和`corpus.ch`中，每行一个句子。
- 训练分词模型：`./tokenizer/tokenize.py`中调用了sentencepiece.SentencePieceTrainer.Train()方法，利用`corpus.en`和`corpus.ch`中的语料训练分词模型，训练完成后会在`./tokenizer`文件夹下生成`chn.model`，`chn.vocab`，`eng.model`和`eng.vocab`，其中`.model`和`.vocab`分别为模型文件和对应的词表。


## Requirements

This repo was tested on Python 3.6+ and PyTorch 1.5.1. The main requirements are:

- tqdm
- pytorch >= 1.5.1
- sacrebleu >= 1.4.14
- sentencepiece >= 0.1.94

To get the environment settled quickly, run:

```
pip install -r requirements.txt
```

## Usage

模型参数在`config.py`中设置。

- 由于transformer显存要求，支持MultiGPU，需要设置`config.py`中的`device_id`列表以及`main.py`中的`os.environ['CUDA_VISIBLE_DEVICES']`。

如要运行模型，可在命令行输入：

```
python main.py
```

实验结果在`./experiment/train.log`文件中，测试集翻译结果在`./experiment/output.txt`中。

如要运行test模型，可在命令行输入：

```
python main.py --model_path $pretrained_path$ --cuda_visible_devices=1 --mode=translate --beam_search 
```

具体超参数请参考./model.py

## One Sentence Translation

将训练好的model或者上述Pretrained model以`model.pth`命名，保存在`./experiment`路径下。在`main.py`中运行`translate_example`，即可实现单句翻译。

如英文输入单句为：

```
The near-term policy remedies are clear: raise the minimum wage to a level that will keep a fully employed worker and his or her family out of poverty, and extend the earned-income tax credit to childless workers.
```

ground truth为：

```
近期的政策对策很明确：把最低工资提升到足以一个全职工人及其家庭免于贫困的水平，扩大对无子女劳动者的工资所得税减免。
```

beam size = 3的翻译结果为：

```
短期政策方案很清楚:把最低工资提高到充分就业的水平,并扩大向无薪工人发放所得的税收信用。
```
