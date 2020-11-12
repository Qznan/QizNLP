#!/usr/bin/env python
# coding=utf-8

import re
import numpy as np

INF = 1e7
zh_char = re.compile('[\u4e00-\u9fa5]')
en_char = re.compile('[a-zA-Z!~?.\s]')
punc_char = re.compile('[,，.。!！?？]]')

bert_embed = None
w2i = None
word_count_prob = None

""" bad starts """
bad_starts = ['我也是', '不是吧', '是不是', '其实', '不能', '但是', '反正', ' 你不是', '哪里有',
              '因为', '难道', '不要', '可是我', '那就不用', '你是不是',
              '不是', '为什么']
# bad_starts += ['你不会','你不是']

""" bad ends """
bad_ends = ['我是你的', '是什么?', '是什么', '不喜欢,喜欢']
bad_ends += ['?', '吗', '么', '什么东西']
# bad_ends += ['<unf>']  # 在限定的长度中未解码完成
# bad_ends += ['的','不会']

""" bad words """
bad_words = [',但是', '小三', '看到你的评论我笑了', '大家都是这样', '我们都是好孩子',
             '下次见面就是你了', '现在玩游戏都是一样的']

"""
re notes
匹配以abc开头: ^(?=abc).*$
不以abc开头: ^(?!abc).*$
以abc结尾: ^.*?(?<=abc)$
不以abc结尾: ^.*?(?<!abc)$
不包含abc: ^(?!.*abc).*$
包含abc: ^.*?abc.*$
"""
bad_re_patterns = ['(.+?)(不是|就是|的|,)\\1',
                   '.{2}的.{2}的',
                   '你(.{2,})我\\1',
                   '没有(.{2,}),只有\\1',
                   '我是(.{2,}),你是\\1',
                   '是(.{1,}),不是\\1',
                   '你才是(.{1,}),我是\\1',
                   '不会(.{1,}),我是\\1',
                   '就是(.{2,})了才会\\1',
                   '.*不(.{1,2}).*(也|要)\\1.*',
                   '.*(?<!不想)(.{1,}).*不想\\1.*',
                   '.*(?<!不)喜欢(.{1,}).*不喜欢\\1.*',
                   '([^哈!~。.])\\1{2,}$',
                   '^(.{2,})[^,!?]+?\\1$',
                   '^(.{2,}).+?\\1(了|的)$',
                   '^.+?\?.+?$',  # 问号在句子中间
                   ]
bad_re_patterns = [re.compile(pattern) for pattern in bad_re_patterns]

# 两个句子中的冲突词，若是包含该冲突词需保证长的在后
# 用逗号分开的两句
conflict_items_1 = [('喜欢', '不喜欢'), ('想', '不想'), ('你们', '我们'), ('想', '不能')]
# q-a对
conflict_items_2 = [('早', '晚'),  # 两句话中任意一句有 早 而另外一句仅有晚
                    ('', '问'),  # 前面一句没有 问 后面一句有 问
                    ('', '不好'),
                    ]


def rerank(results, input_sent = '你'):
    """
    result: answer and score which to be rank. e.g. [['我也想吃', -2.1],['我也想要', -2.5],...]
    input_sent[option]: question
    return: (same struction as result) e.g. [['我也想吃', -0.525], ['我也想要', -0.625]]
    """
    assert results
    if isinstance(results[0], tuple):
        results = [list(item) for item in results if item[0]]  # tuple change to list
    # print(results)
    # print_distinct_ratio(results)

    for i, [output, score] in enumerate(results):
        # bad_case直接分数设为极小值并跳过
        if is_bad(output) \
        or levenshtein_distance(output, input_sent) < 2 \
        or distinct_ratio(output) <= 0.66 \
        or len(output) >= 14 \
        or is_bad1(output) \
        or is_conflict(input_sent, output, conflict_items_2):
            results[i][1] = -INF
            continue
        # 如果问题是英文而恢复全是英文则直接分数设为极小值并跳过
        if not is_english(input_sent) and is_english(output):
            results[i][1] = -INF
            continue
        # 长度归一化
        if is_english(input_sent):  # 如果全是英文正常归一化
            results[i][1] = results[i][1] / len(results[i][0])
        else:
            results[i][1] = results[i][1] / count_length_score(results[i][0])

        # 不同字比率
        # results[i][1] = results[i][1] + distinct_ratio(output) * 0.5

    # # 鼓励不同首字
    # head_freqs = {}
    # for i, [resp, score] in enumerate(results):
    #     if score == -INF:
    #         continue
    #     head = resp[0]
    #     results[i][1] = score - (head_freqs.get(head, 0.) * 0.5)
    #     head_freqs[head] = head_freqs.get(head, 0.) + 1

    # 排序
    rarank_results = sorted(results, key=lambda item: item[1], reverse=True)
    # print('after sorted:', results)
    return rarank_results


def is_bad(sent):
    """ 处理单词包含/起始/结尾的句式
        处理正则匹配命中的句式
    """
    for begin in bad_starts:
        if sent.startswith(begin):
            return True
    for end in bad_ends:
        if sent.endswith(end):
            return True
    for word in bad_words:
        if word in sent:
            return True
    for bad_re_pattern in bad_re_patterns:
        if re.search(bad_re_pattern, sent):
            return True
    return False


def is_bad1(sent):
    """ 处理逗号分隔的句式 """
    if ',' not in sent:
        return False
    seg = sent.split(',')
    if len(seg) > 2:  # 有两个以上逗号
        return False
    s1, s2 = seg
    if not len(s1) or not len(s2):  # 以逗号开头和结尾
        return True
    set1, set2 = set(s1), set(s2)
    if len(s1) >= 4 and len(set1) / len(s1) < 0.6:
        return True
    if len(s2) >= 4 and len(set2) / len(s2) < 0.6:
        return True
    union_set = set1 & set2
    if len(union_set) >= 3:  # 相同字数3个及以上
        return True
    if len(union_set) / len(s1) > 0.5 or len(union_set) / len(s2) > 0.5:  # 前(后)半句几乎仅仅是相同的字
        return True
    if is_conflict(s1, s1, conflict_items_1):
        return True
    return False


def is_conflict(s1, s2, conflict_items):
    """ 两句话前后出现互相矛盾 """
    for i1, i2 in conflict_items:
        if not i1 and i2:  # ('','问')
            if i2 not in s1 and i2 in s2:
                return True
        elif i1 and not i2:  # ('好','')
            if i1 in s1 and i1 not in s2:
                return True
        elif i1 in i2:  # 包含情况，长的是i2 ('喜欢','不喜欢')
            if i1 in s1 and i2 not in s1 and i2 in s2:  # 我喜欢你，我不喜欢你
                return True
            if i1 in s2 and i2 not in s2 and i2 in s1:  # 我不喜欢你，我喜欢你
                return True
        else:  # 不包含的情况 (早,晚) (我们,你们)
            if (i1 in s1 and i2 in s1) or (i1 in s2 and i2 in s1):  # 如果矛盾的两字同时出现在前半句或后半句中,不在此处理
                return False
            if (i1 in s1 and i2 in s2) or (i1 in s2 and i2 in s1):
                return True
    return False


def distinct_ratio(sent):
    sent = re.sub(punc_char, '', sent)
    if len(sent) == 0:
        return 1.
    return round(len(set(sent)) / len(sent), 4)


def is_english(sent):
    if not sent:
        return False
    if len(en_char.sub('', sent)) == 0:
        return True
    else:
        return False


def count_length_score(sent):
    punc = ",。.。!！?？"  # 标点符号
    special_zh_char = '哈呵'  # 特殊字符
    length_score = 0
    for s in sent:
        if re.match(zh_char, s):  # 是中文
            length_score += 1
        elif s in punc:  # 标点符号
            length_score += 0.5
        elif s in special_zh_char:  # 中文特殊符号
            length_score += 0.7
        elif s.isdigit():  # 是数字
            length_score += 0.8
        else:
            pass
    return length_score


def cos_sim(q, a):
    q, a = np.mat(q), np.mat(a)
    num = float(q * a.T)
    denom = np.linalg.norm(q) * np.linalg.norm(a)
    cos = num / denom
    sim = 0.5 + 0.5 * cos  # 归一化为0-1
    return sim


def calc_cos_sim(q, a, embed, w2i):
    vec_q = np.zeros(embed.shape[1])
    vec_a = np.zeros(embed.shape[1])
    for c in q:
        if c in w2i:
            vec_q += embed[w2i[c]]  # bow
    for c in a:
        if c in w2i:
            vec_a += embed[w2i[c]]  # bow
    return cos_sim(vec_q, vec_a)


def levenshtein_distance(s1, s2):
    """ 编辑距离 """
    rows = len(s1) + 1
    columns = len(s2) + 1
    # 创建矩阵
    matrix = [[0 for j in range(columns)] for i in range(rows)]  # row * column
    for j in range(columns):  # 矩阵第一行
        matrix[0][j] = j
    for i in range(rows):  # 矩阵第一列
        matrix[i][0] = i
    # 根据状态转移方程逐步得到编辑距离
    for i in range(1, rows):
        for j in range(1, columns):
            if s1[i - 1] == s2[j - 1]:
                cost = 0  # 不需更改
            else:
                cost = 1  # 替换操作
            matrix[i][j] = min(matrix[i - 1][j - 1] + cost,
                               matrix[i - 1][j] + 1,
                               matrix[i][j - 1] + 1)
    return matrix[rows - 1][columns - 1]


def print_distinct_ratio(output_sents):
    distinct_ratio_list = [[sent, distinct_ratio(sent)] for sent, _ in output_sents]
    distinct_ratio_list.sort(key=lambda x: x[1], reverse=True)
    print('-' * 10, 'distinct_ratio', sep='')
    for item in distinct_ratio_list:
        print(item)
    print('-' * 10, 'distinct_ratio', sep='')


if __name__ == '__main__':
    print(rerank([['我也想吃', -2.1], ['我也想要', -2.5]]))
    pass
