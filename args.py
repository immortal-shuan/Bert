import argparse


def init_arg_parser():
    arg_parser = argparse.ArgumentParser()
    # 数据预处理时的参数
    arg_parser.add_argument('--vocab_path', default='D:/baseline/bert/config/vocab.txt')  # 词表路径
    arg_parser.add_argument('--vocab_size', default=20000, type=int)  # 词表的大小
    arg_parser.add_argument('--rm_punc', default=False)  # 是否去除标点符号
    arg_parser.add_argument('--add_word', default=False)  # 是否添加词特征
    arg_parser.add_argument('--add_bigram', default=False)  # 是否添加双字符特征
    arg_parser.add_argument('--add_trigram', default=False)  # 是否添加三字符特征

    # 训练时参数


    # 各种文档路径


    # 模型内各种参数
    arg_parser.add_argument('--lr', default=2e-5, type=float)            # 学习率
    arg_parser.add_argument('--emb_size', default=512, type=int)     # 词嵌入的维度
    arg_parser.add_argument('--hidden_size', default=1024, type=int)   # 隐藏层的维度
    arg_parser.add_argument('--labels_num', default=2, type=int)    # 标签的数量
    arg_parser.add_argument('--max_len', default=20, type=int)  # 句子的最大长度
    arg_parser.add_argument('--seg_num', default=2, type=int)  # 输入句子的数量
    arg_parser.add_argument('--layers_num', default=6, type=int)  # 模型的解码器层数
    arg_parser.add_argument('--dropout', default=0.2)  # dropout
    arg_parser.add_argument('--heads_num', default=12, type=int)
    arg_parser.add_argument('--feedforward_size', default=4096, type=int)  # 前向维度

    args = arg_parser.parse_args()
    return args