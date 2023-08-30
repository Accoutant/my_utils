import collections
import numpy as np
import torch
import spacy



def split_train_test(x_data, t_data, seed, rate):
    """
    按照seed和rate来区分训练集和测试集
    :param x_data: 未打乱的特征数据，np.array数组
    :param t_data: 未打乱的标签数据, np.array数组
    :param seed: 种子
    :param rate: 划分比例，eg:0.9代表训练集占0.9
    :return: 返回打乱后的(x_train, t_train), (x_test, t_test)， 均为np.array数组
    """
    shuffled_indices = np.arange(x_data.shape[0])
    np.random.seed(seed)
    np.random.shuffle(shuffled_indices)
    x_data, t_data = x_data[shuffled_indices], t_data[shuffled_indices]
    idx = int(rate*x_data.shape[0])
    x_train = x_data[:idx]
    x_test = x_data[idx:]
    t_train = t_data[:idx]
    t_test = t_data[idx:]
    return (x_train, t_train), (x_test, t_test)


def disrupting_data(data, seed):
    """
    打乱数据
    :param data:未打乱的数据
    :param seed: 种子
    :return: 返回打乱后的data
    """
    shuffled_indices = np.arange(data.shape[0])
    np.random.seed(seed)
    np.random.shuffle(shuffled_indices)
    data = data[shuffled_indices]
    return data


def get_k_fold_data(k, i, x, y):
    """
    获取交叉验证数据(不进行打乱)
    :param k: 折数
    :param i: 选中的验证折数
    :param x: np.array数组
    :param y: np.array数组
    :return:(x_train, y_train), (x_test, y_test), np.array数组
    """
    n_train = x.shape[0]
    num_items = n_train // k
    idx = [num_items * j for j in range(k)]
    if i != k:
        x_test = x[idx[i - 1]:idx[i]]
        y_test = y[idx[i - 1]:idx[i]]
        x_train = np.concatenate((x[:idx[i - 1]], x[idx[i]:]), axis=0)
        y_train = np.concatenate((y[:idx[i - 1]], y[idx[i]:]), axis=0)
    else:
        x_test = x[idx[i - 1]:]
        y_test = y[idx[i - 1]:]
        x_train = x[:idx[i - 1]]
        y_train = y[:idx[i - 1]]
    return (x_train, y_train), (x_test, y_test)


def get_time_steps(data, num_steps, jump=True):
    """
    对时间序列按时间步生成一条条数据
    :param data: 未分割的时间序列数据,np.array数组
    :param num_steps: 时间步
    :param jump: 是否间隔排序
    :return: 未打乱的data, np.array数组
    """
    if jump:
        steps = len(data) // num_steps
        data = [data[i*num_steps:(i+1)*num_steps] for i in range(steps)]
        data = np.array(data)
    else:
        n = data.shape[0]
        data = [data[i:i+num_steps] for i in range(n-num_steps+1)]
        data = np.array(data)
    return data


class Tokenizer:
    def __init__(self, min_freq=0, reserved_tokens=None, padding=False,
                 trucation=False, pad_token='<pad>', max_length=None,
                 post_process=False, start_token='<cls>', end_token='<seq>'):
        """
        对文本进行分词并获得对应的vocab， 可以对文本进行编码和解码
        :param min_freq: 最小词频
        :param reserved_tokens: 指定词
        :param padding:是否进行填充
        :param trucation:是否进行截断
        :param pad_token:填充词，默认为'<pad>'
        :param max_length:填充和截断的最大长度
        :param post_process:是否进行位置标注
        :param start_token:位置标注的开始词
        :param end_token:位置标注的结束词
        """
        self.text = None
        self.min_freq = min_freq
        if reserved_tokens is None:
            self.reserved_tokens = []
        else:
            self.reserved_tokens = reserved_tokens
        self.token_to_id = None
        self.id_to_token = None
        self.pad_token = pad_token
        self.max_length = max_length
        self.padding = padding
        self.trucation = trucation
        self.post_process = post_process
        self.start_token = start_token
        self.end_token = end_token
        if self.post_process and self.start_token not in self.reserved_tokens:    # 当不对位置标注时，不加入开始和结束词
            self.reserved_tokens.append(self.start_token)
        if self.post_process and self.end_token not in self.reserved_tokens:      # 当不对位置标注时，不加入开始和结束词
            self.reserved_tokens.append(self.end_token)

    def get_from_text(self, text: list):
        """将句子或者是段落文本进行分词"""
        assert isinstance(text, list)
        if isinstance(text[0], list):
            self.text = [self.__get_token(sentence) for paragraph in text for sentence in paragraph]
        else:
            self.text = [self.__get_token(sentence) for sentence in text]
        return self.text

    def __get_token(self, sentence: str):
        """获得分词列表"""
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(sentence)
        token = [token.text for token in doc]
        return token

    def get_vocab(self):
        if isinstance(self.text, list):
            # 计算计数器
            if isinstance(self.text[0], list):
                all_vocab = [token for sentence in self.text for token in sentence]
                collect = collections.Counter(all_vocab)
            else:
                assert isinstance(self.text[0], str)
                all_vocab = [token for token in self.text]
                collect = collections.Counter(all_vocab)
            # 初始化id_to_token和token_to_id列表
            self.id_to_token = self.reserved_tokens + ['<unk>']
            self.token_to_id = {}
            for token, freq in collect.items():
                if freq < self.min_freq:
                    pass
                else:
                    self.id_to_token.append(token)
            for idx, token in enumerate(self.id_to_token):
                self.token_to_id[token] = idx
        else:
            print('self.text need class of list')

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            id = self.token_to_id.get(tokens, 'unk')
        else:
            id = [self.token_to_id.get(token, 'unk') for token in tokens]
        return id

    def __len__(self):
        return len(self.id_to_token)

    def encode(self, sentence: str):
        """编码"""
        assert isinstance(sentence, str)
        tokens = self.__get_token(sentence)
        # 进行位置标注
        if self.post_process:
            tokens = self.__post_process(tokens)
        # 进行填充
        if self.padding and len(tokens) < self.max_length:
            ids = self.__getitem__(tokens + [self.pad_token] * (self.max_length - len(tokens)))
        else:
            ids = self.__getitem__(tokens)
        # 进行截断
        if self.trucation and len(ids) > self.max_length:
            if self.post_process:
                ids = ids[:self.max_length-1] + [ids[-1]]
            else:
                ids = ids[:self.max_length]
        return ids

    def decode(self, ids: list):
        """解码"""
        assert isinstance(ids, list)
        tokens = [self.id_to_token[id] for id in ids]
        if self.padding:
            tokens = [token for token in tokens if token != self.pad_token]
        if self.post_process:
            tokens.remove(self.start_token)
            tokens.remove(self.end_token)
        text = ' '.join(tokens)
        return text

    def __post_process(self, tokens):
        """标记起始位置"""
        tokens = [self.start_token] + tokens + [self.end_token]
        return tokens








