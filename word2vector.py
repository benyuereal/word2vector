from collections import Counter
from operator import itemgetter as _itemgetter

import numpy as np
from sklearn import preprocessing

import file_interface
import jieba


class word2vector():
    def __init__(self, vector_length=15000, learn_rate=0.025, window_length=5, model='cbow'):
        # 再定义其他变量
        # 这个是去掉停用词之后的词顺序表
        self.model = model
        self.cut_text_list = None
        self.vector_length = vector_length
        self.learn_rate = learn_rate
        self.window_length = window_length
        # 这个是字典 字典里面存储的是该单词的哈夫曼编码、出现的频率、出现的频率
        self.word_dictionary = None  # each element is a dict, including: word,possibility,vector,huffmancode
        # 哈夫曼树
        self.huffman_tree = None

    # 构造训练模型
    # fixme 大体思路:切分单词统计出词频（包括去掉部分单词）、训练模型、保存模型、进行预测
    def train_model(self, text_list):
        if self.huffman_tree == None:
            print(self.word_dictionary)
            if self.word_dictionary == None:
                # 统计出词频
                word_counts = word_count(text_list)
                self.cut_text_list = word_counts.text_list
                # 构造字典
                self.generate_word_dictionary(word_counts.count_result.larger_than(1))
            # 产生一个全量的哈夫曼树
            self.huffman_tree = huffman_tree(self.word_dictionary, vector_length=self.vector_length)
        # 以上，最重要的数据环节 哈夫曼树、单词词频已经产生好了，接下来进行训练
        print('word_dict and huffman tree already generated, ready to train vector')
        # 设置待训练的单词在窗口中左右两边的单词长度,一般情况下窗口是奇数，比如 11 左边是5 右边也是5

        left = (self.window_length - 1) >> 1
        right = self.window_length - 1 - left

        if self.cut_text_list:
            # if the text has been cutted
            total = self.cut_text_list.__len__()
            count = 0
            for line in self.cut_text_list:
                line_len = line.__len__()
                for i in range(line_len):
                    self.deal_gram_CBOW(line[i], line[max(0, i - left):i] + line[i + 1:min(line_len, i + right + 1)])
                count += 1
                print('{c} of {d}'.format(c=count, d=total))

        else:
            # if the text has note been cutted
            for line in text_list:
                line = list(jieba.cut(line, cut_all=False))
                line_len = line.__len__()
                for i in range(line_len):
                    # 相当于拿到目标单词左右两边 窗口大小的单词量，有时候刚开始的时候目标单词左边是只有一个或者没有单词的
                    self.deal_gram_CBOW(line[i], line[max(0, i - left):i] + line[i + 1:min(line_len, i + right + 1)])
        print('word vector has been generated')

    def deal_gram_CBOW(self, word, gram_word_list):
        # 如果单词不在字典表里面 那么肯定直接结束掉了
        if not self.word_dictionary.__contains__(word):
            return
        # 当前这个单词的哈夫曼编码
        huffman_code = self.word_dictionary[word]['huffman_code']
        # 暂时不理解这个
        gram_vector_sum = np.zeros([1, self.vector_length])
        #  这个是python的slice notation的特殊用法。
        #
        # a = [0,1,2,3,4,5,6,7,8,9]
        # b = a[i:j] 表示复制a[i]到a[j-1]，以生成新的list对象
        # b = a[1:3] 那么，b的内容是 [1,2]
        # 当i缺省时，默认为0，即 a[:3]相当于 a[0:3]
        # 当j缺省时，默认为len(alist), 即a[1:]相当于a[1:10]
        # 当i,j都缺省时，a[:]就相当于完整复制一份a了
        #
        # b = a[i:j:s]这种格式呢，i,j与上面的一样，但s表示步进，缺省为1.
        # 所以a[i:j:1]相当于a[i:j]
        # 当s<0时，i缺省时，默认为-1. j缺省时，默认为-len(a)-1
        # fixme 所以a[::-1]相当于 a[-1:-len(a)-1:-1]，
        # fixme 也就是从最后一个元素到第一个元素复制一遍。所以你看到一个倒序的东东。

        # 这一个步骤 相当于累加目标词两边的词向量
        for i in range(gram_word_list.__len__())[::-1]:
            item = gram_word_list[i]
            if self.word_dictionary.__contains__(item):
                gram_vector_sum += self.word_dictionary[item]['huffman_vector']
            else:
                gram_word_list.pop(i)

        if gram_word_list.__len__() == 0:
            return
        e = self.__GoAlong_Huffman(huffman_code, gram_vector_sum, self.huffman_tree.root)
        # 词向量更新
        for item in gram_word_list:
            # 词向量更新
            self.word_dictionary[item]['huffman_vector'] += e
            self.word_dictionary[item]['huffman_vector'] = preprocessing.normalize(
                self.word_dictionary[item]['huffman_vector'])

    # 进行沿着哈夫曼树
    def __GoAlong_Huffman(self, huffman_code, input_vector, root):

        node = root
        e = np.zeros([1, self.vector_length])
        # 假如哈夫曼编码是 1001 也就是 '左右右左'。
        # 以下是每个节点
        for level in range(huffman_code.__len__()):
            huffman_charat = huffman_code[level]
            # 判别正类和负类的方法是使用sigmoid函数
            q = self.__Sigmoid(input_vector.dot(node.huffman_vector.T))
            # 梯度公式 ∂L∂xw=∑j=2lw(1−dwj−σ(xTwθwj−1))θwj−1
            grad = self.learn_rate * (1 - int(huffman_charat) - q)
            # e是输出向量 是将目标单词分类的向量 用来更新xw
            e += grad * node.huffman_vector
            # 更新node的内部哈夫曼向量
            node.huffman_vector += grad * input_vector
            # norm：可以为l1、l2或max，默认为l2
            #
            # 若为l1时，样本各个特征值除以各个特征值的绝对值之和
            #
            # 若为l2时，样本各个特征值除以各个特征值的平方之和
            # In [8]: from sklearn import preprocessing
            #    ...: X = [[ 1., -1., 2.],
            #              [ 2., 0., 0.],
            #              [ 0., 1., -1.]]
            #    ...: normalizer = preprocessing.Normalizer().fit(X)#fit does nothing
            #    ...: normalizer
            #    ...:
            # Out[8]: Normalizer(copy=True, norm='l2')
            #
            # In [9]: normalizer.transform(X)
            # Out[9]:
            # array([[ 0.40824829, -0.40824829,  0.81649658],
            #        [ 1.        ,  0.        ,  0.        ],
            #        [ 0.        ,  0.70710678, -0.70710678]])
            node.huffman_vector = preprocessing.normalize(node.huffman_vector)
            # 0 向右边 1 向左边
            if huffman_charat == '0':
                node = node.right
            else:
                node = node.left
        return e

    def __Sigmoid(self, value):
        return 1 / (1 + np.math.exp(-value))

    # 产生单词的字典 这个字典包含 词频、词值、哈夫曼编码、节点θ等信息
    def generate_word_dictionary(self, word_frequent):

        # 如果word_frequent既不是字典也不是顺序表 那么就报错。word_frequent要么是字典要么是顺序表
        if not isinstance(word_frequent, dict) and not isinstance(word_frequent, list):
            raise ValueError('the word freq info should be a dict or list')
        # 字典对象
        word_dictionary = {}
        if isinstance(word_frequent, dict):
            sum_count = sum(word_frequent.values())
            for word in word_frequent:
                temp_dict = dict(
                    word=word,
                    freq=word_frequent[word],
                    possibility=word_frequent[word] / sum_count,
                    huffman_vector=np.random.random([1, self.vector_length]),
                    huffman_code=None
                )
                word_dictionary[word] = temp_dict
        else:
            # 如果word_frequent是顺序表的结构
            freq_list = [x[1] for x in word_frequent]
            sum_count = sum(freq_list)

            for item in word_frequent:
                temp_dict = dict(
                    word=item[0],
                    freq=item[1],
                    possibility=item[1] / sum_count,
                    huffman_vector=np.random.random([1, self.vector_length]),
                    huffman_code=None
                )
                word_dictionary[item[0]] = temp_dict
        self.word_dictionary = word_dictionary


# 定义哈夫曼节点对象
class huffman_node():
    def __init__(self,
                 # 当前叶子节点，保存的单词的值，比如说单词'this'
                 value,
                 # 哈夫曼节点 当时叶子节点的时候 会有当前词的频率
                 huffman_possibility,
                 # 默认不是叶子节点
                 is_leaf=False,
                 ):
        # 定义哈夫曼当前节点是从上个节点的左拐还是右拐
        self.huffman_direction = None
        # 定义哈夫曼节点对象中的向量θ
        self.huffman_vector = None
        # 定义哈夫曼树的左子树和右子树
        self.left = None
        self.right = None
        # 定义是否是叶子节点
        self.is_leaf = is_leaf
        # 定义哈夫曼节点对象中的哈夫曼编码
        self.huffman_code = ''
        self.value = value
        self.huffman_possibility = huffman_possibility


# 利用了归并排序
def merge_sort_node_list(node_list, first, end):
    # 其实就是寻找这个顺序表里面possibility最小的那个
    while node_list.__len__() > 1:
        if first >= end:
            return
            # 计算中间索引位置
        mid = int((first + end) / 2)
        merge_sort_node_list(node_list, first, mid)
        merge_sort_node_list(node_list, mid, end)
        merge(node_list, first, mid, end)


def merge(node_list, first, mid, end):
    temp = []
    # 列表前半部分所以位置
    i = first
    # 列表后半部分索引位置
    j = mid + 1
    # 哪个部分小 先放哪个
    while i <= mid and j <= end:
        if node_list[i].possibility <= node_list[j].possibility:
            temp.append(node_list[i])
            i += 1
        else:
            temp.append(node_list[j])
            j += 1
    # 左半部分 或者 右半部分没有放干净的都放置完
    while i <= mid:
        temp.append(node_list[i])
        i += 1
    while j <= end:
        temp.append(node_list[j])
        j += 1
    # 最后将这部分操作的用有序的temp替换掉
    for k in range(0, len(temp)):
        node_list[first + k] = temp[k]


class huffman_tree():
    def __init__(self,
                 # 定义哈夫曼树

                 # 然后有一个保存叶子节点的字典表
                 word_dictionary,
                 vector_length=15000,
                 # 然后有一个代表着当前的单词个数
                 ):

        self.root = None

        # 下面有几个值是需要在传入初始化变量进行计算的
        # 首先是根据传入的字典表 构造成一个字典顺序表
        word_dictionary_list = list(word_dictionary.values())
        self.word_dictionary_list = word_dictionary_list
        # 然后是向量个数
        self.vector_length = vector_length
        # 然后将每个节点都初始化放进去
        node_list = [huffman_node(x['word'], x['possibility'], True) for x in word_dictionary_list]
        # 以上，算是将各个节点的放进顺序表里面，下面还会产生哈夫曼结构、哈夫曼编码
        # self.build_huffman_root(node_list)
        # 组装一个哈夫曼树形结构
        self.build_huffman_tree(node_list)
        # 产生哈夫曼编码
        self.generate_huffman_code(self.root, word_dictionary)

    # 组装一个哈夫曼树根
    def build_huffman_root(self,
                           # 需要传入的值 以下是
                           # 传入的是一个节点顺序表
                           node_list
                           ):
        first = 0
        end = node_list.__len__() - 1
        # 其实就是寻找这个顺序表里面possibility最小的那个
        while node_list.__len__() > 1:
            if first >= end:
                return
                # 计算中间索引位置
            merge_sort_node_list(node_list, first, end)
        # 将root赋值 也就是node_list的第一个索引位置
        self.root = node_list[0]

    # 组装哈夫曼树
    # 现在的node_list只是按照possibility来排序的一个顺序表，下面是将这个node_list加上节点
    # 思路：每两个node作为一个节点
    def build_huffman_tree(self, node_list):
        current_node_list = []
        node_list.sort(key=lambda x: x.huffman_possibility, reverse=True)

        while len(node_list) != 1:
            # 最小的
            minimal = node_list[-1]
            # 次小的
            minor = node_list[-2]
            root = self.merge_left_right_and_root(minimal, minor)
            # 将已经使用过的树的部分删除掉
            node_list.pop(-1)
            node_list.pop(-1)
            node_list.append(root)
            # 按照节点的possibility进行重新排序
            node_list.sort(key=lambda x: x.huffman_possibility, reverse=True)
        # 倒数第一个 也仅有一个是树
        self.root = node_list[-1]

    # 产生哈夫曼编码 利用递归
    def generate_huffman_code(self, root, word_dictionary):

        # 到叶子节点就结束了
        if root.is_leaf is True:
            word_dictionary[root.value]['huffman_code'] = root.huffman_code
            return
        else:
            # 根节点
            # TODO 这里用到的就是 哈夫曼编码
            code = root.huffman_code
            # 如果是最上面的根节点 那么它的huffman是code
            left_node = root.left
            # 左边是 1 右边是 0
            if left_node != None:
                left_code = code + '1'
                left_node.huffman_code = left_code
                # 然后再判定一下 如果左 或者 右 是叶子 那么久赋值
                self.generate_huffman_code(left_node, word_dictionary)
            right_node = root.right
            if right_node != None:
                right_code = code + '0'
                right_node.huffman_code = right_code
                # 只有叶子节点在字典表里面才能找得到
                self.generate_huffman_code(right_node, word_dictionary)

    # 构造出当前这一对叶子节点与根节点的数对
    def merge_left_right_and_root(self, node1, node2):
        root_possibility = node1.huffman_possibility + node2.huffman_possibility
        root = huffman_node(None, root_possibility, False)
        # 初始化θ
        root.huffman_vector = np.zeros([1, self.vector_length])
        if node1.huffman_possibility > node2.huffman_possibility:
            # 大的都放左边
            root.left = node1
            root.right = node2
        else:
            root.left = node2
            root.right = node1
        return root


# 词频统计类 主要功能有 切分词、统计词频
class word_count():
    # can calculate the freq of words in a text list

    # for example
    # >>> data = ['Merge multiple sorted inputs into a single sorted output',
    #           'The API below differs from textbook heap algorithms in two aspects']
    # >>> wc = WordCounter(data)
    # >>> print(wc.count_res)

    # >>> MulCounter({' ': 18, 'sorted': 2, 'single': 1, 'below': 1, 'inputs': 1, 'The': 1, 'into': 1, 'textbook': 1,
    #                'API': 1, 'algorithms': 1, 'in': 1, 'output': 1, 'heap': 1, 'differs': 1, 'two': 1, 'from': 1,
    #                'aspects': 1, 'multiple': 1, 'a': 1, 'Merge': 1})

    def __init__(self, text_list):
        self.text_list = text_list
        stop_word = self.stop_word();
        self.stop_word = stop_word
        self.count_result = None
        # 进行词频统计
        self.word_count(self.text_list)

    def stop_word(self):
        stop_words = file_interface.load_pickle('./static/stop_words.pkl')
        return stop_words

    # 单词切分 去掉停用词
    def word_count(self, text_list, cut_all=False):
        # 过滤后的单词顺序表
        filtered_word_list = []
        count = 0
        #
        for line in text_list:
            result = jieba.cut(line, cut_all=cut_all)
            result = list(result)
            text_list[count] = result
            count += 1
            filtered_word_list += result
        # 这个属性需要留意一下 fixme
        self.count_result = mul_counter(filtered_word_list)
        # 过滤掉停用词
        for word in self.stop_word:
            try:
                self.count_result.pop(word)
            except:
                pass


# 进行词频统计的类
class mul_counter(Counter):
    def __init__(self, element_list):
        super().__init__(element_list)

    # minvalue 是这个单词最低出现的频率
    def larger_than(self, minvalue, ret='list'):
        temp = sorted(self.items(), key=_itemgetter(1), reverse=True)
        low = 0
        high = temp.__len__()
        while (high - low > 1):
            mid = (low + high) >> 1
            if temp[mid][1] >= minvalue:
                low = mid
            else:
                high = mid
        if temp[low][1] < minvalue:
            if ret == 'dict':
                return {}
            else:
                return []
        if ret == 'dict':
            ret_data = {}
            for ele, count in temp[:high]:
                ret_data[ele] = count
            return ret_data
        else:
            return temp[:high]

    def less_than(self, maxvalue, ret='list'):
        temp = sorted(self.items(), key=_itemgetter(1))
        low = 0
        high = temp.__len__()
        while ((high - low) > 1):
            mid = (low + high) >> 1
            if temp[mid][1] <= maxvalue:
                low = mid
            else:
                high = mid
        if temp[low][1] > maxvalue:
            if ret == 'dict':
                return {}
            else:
                return []
        if ret == 'dict':
            ret_data = {}
            for ele, count in temp[:high]:
                ret_data[ele] = count
            return ret_data
        else:
            return temp[:high]


if __name__ == '__main__':
    # text = FI.load_pickle('./static/demo.pkl')
    # text =[ x['dealed_text']['left_content'][0] for x in text]

    data = [
        'Cristiano Ronaldo dos Santos Aveiro (Cristiano Ronaldo dos Santos Aveiro), born on February 5, 1985 in Funchal, Madeira, Portugal, is a professional football player in Portugal. The winger and center play for Juventus Football Club of Italy and are the captain of the Portuguese national men soccer team.Cristiano Ronaldo debuted at Sporting Lisbon. In 2003, he joined the Premier League Manchester United. He won 10 championships in the English Premier League, the UEFA Champions League and the World Club Cup. In June 2009, he was transferred to Real Madrid with a price of 96 million euros. He won 16 championships in the UEFA Champions League, 2 La Liga titles and 3 World Cup Cup championships. Cristiano Ronaldo played for Real Madrid in 9 years, contributing 450 goals and 131 assists in 438 games, and became the player with the highest scoring rate in Real Madrid history with a scoring rate of 1.03 goals per game.Cristiano Ronaldo maintains a number of personal records, including the total scores of individual scores in the five major leagues in Europe, the total score of the Real Madrid club individual goals, the total score of the Champions League individual goals, and the total scores of the European national team individual goals. Cristiano Ronaldo has won the Golden Globe Award five times, three times to win the World Footballer, four times to win the European Golden Boot Award, and seven times to win the Champions League top scorer and other personal honors.In July 2016, C Ronaldo led Portugal to the 2016 European Cup in France, the first international competition in the history of the Portuguese national team. On July 10, 2018, the transfer to Serie A Juventus. On July 24, 2018, Cristiano Ronaldo was selected as the candidate for the 2018 FIFA World Footballer. In September 2018, he was selected as the FIFA Team of the Year.']
    word2vec = word2vector(vector_length=500)
    word2vec.train_model(data)
    file_interface.save_pickle(word2vec.word_dictionary, './static/word2vector.pkl')

    data = file_interface.load_pickle('./static/word2vector.pkl')
    x = {}
    for key in data:
        temp = data[key]['huffman_vector']
        temp = preprocessing.normalize(temp)
        x[key] = temp
    file_interface.save_pickle(x, './static/normal_word2vector.pkl')

    x = file_interface.load_pickle('./static/normal_word2vector.pkl')


    def cal_simi(data, key1, key2):
        return data[key1].dot(data[key2].T)[0][0]


    possibility_list = []
    keys = list(x.keys())
    for key in keys:
        every = huffman_node(
            key,cal_simi(x, 'United', key))
        possibility_list.append(every)

    possibility_list.sort(key=lambda x: x.huffman_possibility, reverse=True)

    for obj in possibility_list:
        print(obj.value, '\t', obj.huffman_possibility)

