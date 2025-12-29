import json

if __name__ == "__main__":
    files = ['train_10k', 'train_100k', 'test','valid']
    ch_path = 'corpus.ch'
    en_path = 'corpus.en'
    ch_lines = []
    en_lines = []

    for file in files:
        file_path = f'./json/{file}.jsonl'  # 后缀改为jsonl，若你的文件仍用.json后缀可保留，仅改读取逻辑
        try:
            # JSONL格式：逐行读取，每行解析为独立的JSON对象
            with open(file_path, 'r', encoding='utf-8') as f:
                line_num = 0
                for line in f:
                    line_num += 1
                    line = line.strip()  # 去除换行/空格，避免空行解析报错
                    if not line:  # 跳过空行
                        continue
                    try:
                        item = json.loads(line)  # 逐行解析JSON，而非一次性加载整个文件
                        # 校验en/zh键是否存在
                        if 'en' in item and 'zh' in item:
                            en_lines.append(item['en'] + '\n')
                            ch_lines.append(item['zh'] + '\n')
                        else:
                            print(f"警告：{file_path} 第{line_num}行缺少'en'或'zh'键，已跳过")
                    except json.JSONDecodeError:
                        print(f"错误：{file_path} 第{line_num}行不是合法的JSON格式，已跳过")
        
        except FileNotFoundError:
            print(f"错误：未找到文件 {file_path}")

    # 写入语料文件（保持utf-8编码）
    with open(ch_path, "w", encoding='utf-8') as fch:
        fch.writelines(ch_lines)

    with open(en_path, "w", encoding='utf-8') as fen:
        fen.writelines(en_lines)

    # 输出统计信息
    print("lines of Chinese: ", len(ch_lines))
    print("lines of English: ", len(en_lines))
    print("-------- Get Corpus ! --------")