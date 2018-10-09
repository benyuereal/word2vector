from collections import Counter
from operator import itemgetter as _itemgetter

import numpy as np
import file_interface
import jieba



class word2vector():
    def __init__(self,
                 # 下面是各种变量
                 vector_length=15000,
                 # 学习率
                 learn_rate=0.025,
                 # 窗口大小
                 window_length=5,

                 ):
        # 再定义其他变量
        # 这个是去掉停用词之后的词顺序表
        self.cut_text_list = None,
        self.vector_length = vector_length,
        self.learn_rate = learn_rate,
        self.window_length = window_length,
        # 这个是字典 字典里面存储的是该单词的哈夫曼编码、出现的频率、出现的频率
        self.word_dictionary = None,
        # 默认哈夫曼编码是 ''
        self.huffman = ''

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
        self.huffman_direction = None,
        # 定义哈夫曼节点对象中的向量θ
        self.huffman_vector = None,
        # 定义哈夫曼树的左子树和右子树
        self.left = None,
        self.right = None,
        # 定义是否是叶子节点
        self.is_leaf = is_leaf,
        # 定义哈夫曼节点对象中的哈夫曼编码
        self.huffman_code = None,
        self.value = value
        self.huffman_possibility = huffman_possibility,


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
                 # 首先有个根节点
                 root,
                 # 然后有一个保存叶子节点的字典表
                 word_dictionary,
                 vector_length=15000,
                 # 然后有一个代表着当前的单词个数
                 word_length=None,

                 ):

        # 下面有几个值是需要在传入初始化变量进行计算的
        # 首先是根据传入的字典表 构造成一个字典顺序表
        word_dictionary_list = list(word_dictionary.values)
        self.word_dictionary_list = word_dictionary_list
        # 然后是向量个数
        self.vector_length = vector_length
        # 然后将每个节点都初始化放进去
        node_list = [huffman_node(x['word'], x['possibility'], True) for x in word_dictionary_list]
        # 以上，算是将各个节点的放进顺序表里面，下面还会产生哈夫曼结构、哈夫曼编码
        self.build_huffman_root(node_list)
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
        end = node_list.__len__()
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

        while len(node_list) != 1:
            # 最小的
            minimal = node_list[0]
            # 次小的
            minor = node_list[1]
            root = self.merge_left_right_and_root(minimal, minor)
            node_list.append(root)
            # 将已经使用过的树的部分删除掉
            node_list.pop(0)
            node_list.pop(0)
            # 按照节点的possibility进行重新排序
            node_list.sort(key=lambda x: x.possibility, reverse=True)
        # 倒数第一个 也仅有一个是树
        self.root = node_list[-1]

    # 产生哈夫曼编码 利用递归
    def generate_huffman_code(self, root, word_dictionary):

        if root is None:
            return
        else:
            # 如果当前的点不是叶子节点
            # 根节点
            while root.is_leaf is False:
                code = root.huffman
                # 如果是最上面的根节点 那么它的huffman是code
                left_node = root.left
                # 左边是 1 右边是 0
                if (left_node != None):
                    left_code = code + '1'
                    left_node.huffman = left_code
                    word_dictionary[left_node.value]['huffman'] = left_code
                    # 然后再判定一下 如果左 或者 右 是叶子 那么久赋值
                    self.generate_huffman_code(left_node)
                right_node = root.right
                if (right_node != None):
                    right_code = code + '0'
                    right_node.huffman = right_code
                    word_dictionary[right_node.value]['huffman'] = right_code
                    self.generate_huffman_code(right_node)

    # 构造出当前这一对叶子节点与根节点的数对
    def merge_left_right_and_root(self, node1, node2):
        root_possibility = node1.possibility + node2.possibility
        root = huffman_node(None, root_possibility, False)
        # 初始化θ
        root.huffman_vector = np.zeros([1, self.vector_length])
        if node1.possibility > node2.possibility:
            # 大的都放左边
            root.left = node1
            root.right = node2
        else:
            root.left = node2
            root.right = node1
        return root


# 词频统计类 主要功能有 切分词、统计词频
class word_count():

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
    def word_count(self,text_list,cut_all=False):
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
    def __init__(self,element_list):
        super().__init__(element_list)

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

