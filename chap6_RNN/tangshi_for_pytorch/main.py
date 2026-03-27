import numpy as np
import collections
import torch
from torch.autograd import Variable
import torch.optim as optim

import rnn

start_token = 'G'
end_token = 'E'
batch_size = 64


def process_poems1(file_name):
    """

    :param file_name:
    :return: poems_vector  have tow dimmention ,first is the poem, the second is the word_index
    e.g. [[1,2,3,4,5,6,7,8,9,10],[9,6,3,8,5,2,7,4,1]]

    """
    poems = []
    with open(file_name, "r", encoding='utf-8', ) as f:
        for line in f.readlines():
            try:
                title, content = line.strip().split(':')
                # content = content.replace(' ', '').replace('，','').replace('。','')
                content = content.replace(' ', '')
                if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content or \
                                start_token in content or end_token in content:
                    continue
                if len(content) < 5 or len(content) > 80:
                    continue
                content = start_token + content + end_token
                poems.append(content)
            except ValueError as e:
                print("error")
                pass
    # 按诗的字数排序
    poems = sorted(poems, key=lambda line: len(line))
    # print(poems)
    # 统计每个字出现次数
    all_words = []
    for poem in poems:
        all_words += [word for word in poem]
    counter = collections.Counter(all_words)  # 统计词和词频。
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])  # 排序
    words, _ = zip(*count_pairs)
    words = words[:len(words)] + (' ',)
    word_int_map = dict(zip(words, range(len(words))))
    poems_vector = [list(map(word_int_map.get, poem)) for poem in poems]
    return poems_vector, word_int_map, words

def process_poems2(file_name):
    """
    :param file_name:
    :return: poems_vector  have tow dimmention ,first is the poem, the second is the word_index
    e.g. [[1,2,3,4,5,6,7,8,9,10],[9,6,3,8,5,2,7,4,1]]

    """
    poems = []
    with open(file_name, "r", encoding='utf-8', ) as f:
        # content = ''
        for line in f.readlines():
            try:
                line = line.strip()
                if line:
                    content = line.replace(' '' ', '').replace('，','').replace('。','')
                    if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content or \
                                    start_token in content or end_token in content:
                        continue
                    if len(content) < 5 or len(content) > 80:
                        continue
                    # print(content)
                    content = start_token + content + end_token
                    poems.append(content)
                    # content = ''
            except ValueError as e:
                # print("error")
                pass
    # 按诗的字数排序
    poems = sorted(poems, key=lambda line: len(line))
    # print(poems)
    # 统计每个字出现次数
    all_words = []
    for poem in poems:
        all_words += [word for word in poem]
    counter = collections.Counter(all_words)  # 统计词和词频。
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])  # 排序
    words, _ = zip(*count_pairs)
    words = words[:len(words)] + (' ',)
    word_int_map = dict(zip(words, range(len(words))))
    poems_vector = [list(map(word_int_map.get, poem)) for poem in poems]
    return poems_vector, word_int_map, words

def generate_batch(batch_size, poems_vec, word_to_int):
    n_chunk = len(poems_vec) // batch_size
    x_batches = []
    y_batches = []
    for i in range(n_chunk):
        start_index = i * batch_size
        end_index = start_index + batch_size
        x_data = poems_vec[start_index:end_index]
        y_data = []
        for row in x_data:
            y  = row[1:]
            y.append(row[-1])
            y_data.append(y)
        """
        x_data             y_data
        [6,2,4,6,9]       [2,4,6,9,9]
        [1,4,2,8,5]       [4,2,8,5,5]
        """
        # print(x_data[0])
        # print(y_data[0])
        # exit(0)
        x_batches.append(x_data)
        y_batches.append(y_data)
    return x_batches, y_batches


def run_training():
    # 处理数据集
    # poems_vector, word_to_int, vocabularies = process_poems2('./tangshi.txt')
    poems_vector, word_to_int, vocabularies = process_poems1('poems.txt')
    # 生成batch
    print("finish  loadding data")
    BATCH_SIZE = 100

    torch.manual_seed(5)
    word_embedding = rnn.word_embedding( vocab_length= len(word_to_int) + 1 , embedding_dim= 100)
    rnn_model = rnn.RNN_model(batch_sz = BATCH_SIZE,vocab_len = len(word_to_int) + 1 ,word_embedding = word_embedding ,embedding_dim= 100, lstm_hidden_dim=128)

    # optimizer = optim.Adam(rnn_model.parameters(), lr= 0.001)
    optimizer=optim.RMSprop(rnn_model.parameters(), lr=0.01)

    loss_fun = torch.nn.NLLLoss()
    # rnn_model.load_state_dict(torch.load('./poem_generator_rnn'))  # if you have already trained your model you can load it by this line.

    for epoch in range(30):
        batches_inputs, batches_outputs = generate_batch(BATCH_SIZE, poems_vector, word_to_int)
        n_chunk = len(batches_inputs)
        for batch in range(n_chunk):
            batch_x = batches_inputs[batch]
            batch_y = batches_outputs[batch] # (batch , time_step)

            loss = 0
            for index in range(BATCH_SIZE):
                x = np.array(batch_x[index], dtype = np.int64)
                y = np.array(batch_y[index], dtype = np.int64)
                x = Variable(torch.from_numpy(np.expand_dims(x,axis=1)))
                y = Variable(torch.from_numpy(y ))
                pre = rnn_model(x)
                loss += loss_fun(pre , y)
                if index == 0:
                    _, pre = torch.max(pre, dim=1)
                    print('prediction', pre.data.tolist()) # the following  three line can print the output and the prediction
                    print('b_y       ', y.data.tolist())   # And you need to take a screenshot and then past is to your homework paper.
                    print('*' * 30)
            loss  = loss  / BATCH_SIZE
            print("epoch  ",epoch,'batch number',batch,"loss is: ", loss.data.tolist())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(rnn_model.parameters(), 1)
            optimizer.step()

            if batch % 20 ==0:
                torch.save(rnn_model.state_dict(), './poem_generator_rnn')
                print("finish  save model")



def to_word(predict, vocabs):  # 预测的结果转化成汉字
    # 将输入的 list 转换为 numpy 数组
    predict = np.array(predict)
    
    # 预测结果是 log_softmax 输出，先转回概率分布
    predict = np.exp(predict) 
    # 归一化，确保概率和为 1
    predict = predict / np.sum(predict)
    
    # 【关键修改】：按 predict 的实际长度来抽样，解决 size 不匹配的问题
    sample = np.random.choice(len(predict), p=predict)

    # 越界保护（和原代码逻辑一致）
    if sample >= len(vocabs):
        sample = len(vocabs) - 1

    return vocabs[sample]


def pretty_print_poem(poem):  # 令打印的结果更工整
    shige=[]
    for w in poem:
        if w == start_token or w == end_token:
            break
        shige.append(w)
    poem_sentences = poem.split('。')
    for s in poem_sentences:
        if s != '' and len(s) > 10:
            print(s + '。')


def gen_poem(rnn_model, word_int_map, vocabularies, begin_word):
    poem = begin_word
    word = begin_word
    
    # 为了生成效果，先喂给模型一个开始符 G
    input_ids = [word_int_map[start_token], word_int_map[begin_word]]
    
    while word != end_token:
        input_tensor = Variable(torch.from_numpy(np.array(input_ids, dtype=np.int64)))
        # 如果是 GPU 运行需要 .cuda()
        output = rnn_model(input_tensor, is_test=True)
        
        # 获取预测的下一个字
        word = to_word(output.data.tolist()[-1], vocabularies)
        
        if word == end_token or word == ' ':
            break
            
        poem += word
        input_ids.append(word_int_map[word])
        
        if len(poem) > 50: # 限制长度防止死循环
            break
    return poem



# 1. 训练阶段（如果你已经训练好了，可以注释掉下面这行）
# run_training() 

# 2. 生成阶段
print("\n开始生成诗歌...\n")

# 只加载一次资源
poems_vector, word_int_map, vocabularies = process_poems1('poems.txt')
word_embedding = rnn.word_embedding(vocab_length=len(word_int_map) + 1, embedding_dim=100)
rnn_model = rnn.RNN_model(batch_sz=64, vocab_len=len(word_int_map) + 1, 
                          word_embedding=word_embedding,
                          embedding_dim=100, lstm_hidden_dim=128)

# 加载训练好的权重
rnn_model.load_state_dict(torch.load('./poem_generator_rnn'))
rnn_model.eval() # 开启评估模式

start_words = ["日", "红", "山", "夜", "湖", "湖", "湖", "君"]

for w in start_words:
    p = gen_poem(rnn_model, word_int_map, vocabularies, w)
    print(f"--- 开头词：{w} ---")
    pretty_print_poem(p)
    print()


