#!/usr/bin/env python
# coding=utf-8
import os, re, glob, shutil
import json
import random
import numpy as np
import logging
import logging.handlers
from collections import Counter

try:
    import xlrd, xlwt, openpyxl
except ImportError:
    print('找不到xlrd/xlwt/openpyxl包 不能使用xls相关函数')


def file2list(in_file, strip_nl=True, encoding='U8'):
    with open(in_file, 'r', encoding=encoding) as f:
        lines = [line.strip('\n') if strip_nl else line for line in f]
        print(f'read ok! filename: {in_file}, length: {len(lines)}')
    return lines


def file2txt(in_file, encoding='U8'):
    with open(in_file, 'r', encoding=encoding) as f:
        txt = f.read()
    return txt


# extract: [0,2] or (0,2) or '02'  # assume indices > 9 should not use str
# filter_fn: lambda item: item[2] == 'Y'
def file2items(in_file, strip_nl=True, deli='\t', extract=None, filter_fn=None, encoding='U8'):
    lines = file2list(in_file, strip_nl=strip_nl, encoding=encoding)
    items = [line.split(deli) for line in lines]
    if filter_fn is not None:
        items = list(filter(filter_fn, items))
        print(f'after filter, length: {len(lines)}')
    if extract is not None:
        assert isinstance(extract, (list, tuple, str)), 'invalid extract args'
        items = [[item[int(e)] for e in extract] for item in items]
    return items


# kv_ids: [0,1] or (0,1) or '01'  # assume indices > 9 should not use str
def file2dict(in_file, deli='\t', kv_order='01'):
    items = file2items(in_file, deli=deli)
    assert isinstance(kv_order, (list, tuple, str)) and len(kv_order) == 2, 'invalid kv_order args'
    k_idx = int(kv_order[0])
    v_idx = int(kv_order[1])
    return {item[k_idx]: item[v_idx] for item in items}


# l1,l2,seg_l,l3,l4,l5,seg_l -> [[l1,l2],[l3,l4,l5]]
def file2nestlist(in_file, strip_nl=True, encoding='U8', seg_line=''):
    lst = file2list(in_file, strip_nl=strip_nl, encoding=encoding)
    out_lst_lst = []
    out_lst = []
    for line in lst:
        if line == seg_line:
            out_lst_lst.append(out_lst)
            out_lst = []
            continue
        out_lst.append(line)
    if out_lst:
        out_lst_lst.append(out_lst)
    return out_lst_lst


# [l1,l2,seg,l3,l4,l5] -> [[l1,l2],[l3,l4,l5]]
def seg_list(lst, is_seg_fn=None):
    if is_seg_fn is None:
        is_seg_fn = lambda e: e == ''
    nest_lst = [[]]
    for e in lst:
        if is_seg_fn(ele):
            nest_lst.append([])
            continue
        nest_lst[-1].append(ele)
    return nest_lst


def obj2json(obj, out_file, indent=4):
    def np2py(obj):
        # np格式不支持json序列号,故转化为python数据类型
        if isinstance(obj, (int, float, bool, str)):
            return obj
        if isinstance(obj, dict):
            for k in obj:
                obj[k] = np2py(obj[k])
            return obj
        elif isinstance(obj, (list, tuple)):
            for i in range(len(obj)):
                obj[i] = np2py(obj[i])
            return obj
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    obj = np2py(obj)
    json.dump(obj, open(out_file, 'w', encoding='U8'),
              ensure_ascii=False, indent=indent)
    print(f'save json file ok! {out_file}')


def list_dir_and_file(path):
    lsdir = os.listdir(path)
    dirs = [d for d in lsdir if os.path.isdir(os.path.join(path, d))]
    files = [f for f in lsdir if os.path.isfile(os.path.join(path, f))]
    return dirs, files


def list2file(lines, out_file, add_nl=True, deli='\t'):
    # 兼容
    if isinstance(lines, str):
        lines, out_file = out_file, lines
    assert len(lines) > 0, 'lines must be not None'
    with open(out_file, 'w', encoding='U8') as f:
        if isinstance(lines[0], (list, tuple)):
            lines = [deli.join(map(str, item)) for item in lines]
        # other: str, int, float, bool, obj will use f'{} to strify
        out_list = [f'{line}\n' if add_nl else f'{line}' for line in lines]
        f.writelines(out_list)
        print(f'save ok! filename: {out_file}, length: {len(out_list)}')


def freqs(lst):
    c = Counter(lst)
    return c.most_common()


def list2stats(in_lst, out_file=None):
    stats = freqs(in_lst)
    print(*stats, sep='\n')
    if out_file is not None:
        list2file(stats, out_file, deli='\t')


# sheet_ids: [0,1,3] default read all sheet
def xls2items(in_xls, start_row=1, sheet_ids=None):
    items = []
    xls = xlrd.open_workbook(in_xls)
    sheet_ids = list(range(xls.nsheets)) if sheet_ids is None else sheet_ids
    for sheet_id in sheet_ids:
        sheet = xls.sheet_by_index(sheet_id)
        nrows, ncols = sheet.nrows, sheet.ncols
        print(f'reading... sheet_id:{sheet_id} sheet_name:{sheet.name} rows:{nrows} cols:{ncols}')
        for i in range(start_row, nrows):
            items.append([sheet.cell_value(i, j) for j in range(ncols)])
    return items


# this only support old xls (nrows<65537)
def items2xls_old(items, out_xls=None, sheet_name=None, header=None, workbook=None, max_row_per_sheet=65537):
    workbook = xlwt.Workbook(encoding='utf-8') if workbook is None else workbook
    sheet_name = '1' if sheet_name is None else sheet_name
    num_sheet = 1
    worksheet = workbook.add_sheet(f'{sheet_name}_{num_sheet}')  # 创建一个sheet
    if header is not None:
        for j in range(len(header)):
            worksheet.write(0, j, header[j])
    row_ptr = 1 if header is not None else 0
    for item in items:
        if row_ptr + 1 > max_row_per_sheet:
            num_sheet += 1
            worksheet = workbook.add_sheet(f'{sheet_name}_{num_sheet}')
            if header is not None:
                for j in range(len(header)):
                    worksheet.write(0, j, header[j])
            row_ptr = 1 if header is not None else 0

        for j in range(len(item)):
            worksheet.write(row_ptr, j, item[j])
        row_ptr += 1
    if out_xls is not None:  # 如果为None表明调用者其实还有新的items要加到新的sheet要中，只想要返回的workbook对象
        workbook.save(out_xls)
        print(f'save ok! xlsname: {out_xls}, num_sheet: {num_sheet}')
    return workbook


def items2xls(items, out_xls=None, sheet_name=None, header=None, workbook=None, max_row_per_sheet=65537):
    if workbook is None:
        workbook = openpyxl.Workbook()  # create new workbook instance
        active_worksheet = workbook.active
        workbook.remove(active_worksheet)
    if sheet_name is None:
        sheet_name = 'sheet'
    num_sheet = 1
    worksheet = workbook.create_sheet(f'{sheet_name}_{num_sheet}' if len(items) > max_row_per_sheet else f'{sheet_name}')  # 创建一个sheet
    if header is not None:
        for j in range(len(header)):
            worksheet.cell(0 + 1, j + 1, header[j])  # cell x y 从1开始
    row_ptr = 1 if header is not None else 0
    for item in items:
        if row_ptr + 1 > max_row_per_sheet:
            num_sheet += 1
            worksheet = workbook.create_sheet(f'{sheet_name}_{num_sheet}')
            if header is not None:
                for j in range(len(header)):
                    worksheet.cell(0 + 1, j + 1, header[j])
            row_ptr = 1 if header is not None else 0

        for j in range(len(item)):
            worksheet.cell(row_ptr + 1, j + 1, item[j])
        row_ptr += 1
    if out_xls is not None:  # 如果为None表明调用者其实还有新的items要加到新的sheet表中，只想要返回的workbook对象
        workbook.save(out_xls)
        print(f'save ok! xlsname: {out_xls}, num_sheet: {num_sheet}')
    return workbook


def merge_file(file_list, out_file, shuffle=False):
    assert isinstance(file_list, (list, tuple))
    ret_lines = []
    for i, file in enumerate(file_list):
        lines = file2list(file, strip_nl=False)
        print(f'已读取第{i}个文件:{file}\t行数{len(lines)}')
        ret_lines.extend(lines)
    if shuffle:
        random.shuffle(ret_lines)
    list2file(out_file, ret_lines, add_nl=False)


def merge_file_by_pattern(pattern, out_file, shuffle=False):
    file_list = glob.glob(pattern)
    merge_file(file_list, out_file, shuffle)


# ratio: [18,1,1] or '18:1:1'
# num: 1000
def split_file(file_or_lines, num=None, ratio=None, files=None, shuffle=True, seed=None):
    assert num or ratio, 'invalid args: at least use num or ratio'
    if type(file_or_lines) == str:
        lines = file2list(file_or_lines, strip_nl=False)
    else:
        assert isinstance(file_or_lines, (list, tuple)), 'invalid args file_or_lines'
        lines = file_or_lines
    length = len(lines)
    if shuffle:
        if seed is not None:
            random.seed(seed)
        random.shuffle(lines)
    if num:
        assert num < length, f'invalid args num: num:{num} should < filelen: {length}'
        lines1 = lines[:num]
        lines2 = lines[num:]
        if files:
            assert len(files) == 2
            list2file(files[0], lines1, add_nl=False)
            list2file(files[1], lines2, add_nl=False)
        return lines1, lines2
    if ratio:  # [6,2,2]
        if isinstance(ratio, str):
            ratio = list(map(int, ratio.split(':')))
        cumsum_ratio = np.cumsum(ratio)  # [6,8,10]
        sum_ratio = cumsum_ratio[-1]  # 10
        assert sum_ratio <= length, f'invalid args ratio: ratio:{ratio} should <= filelen: {length}'
        indices = [length * r // sum_ratio for r in cumsum_ratio]  # [6,8,11] if length=11
        indices = [0] + indices
        split_lines_lst = []
        for i in range(len(indices) - 1):
            split_lines_lst.append(lines[indices[i]:indices[i + 1]])
        if files:
            assert len(files) == len(split_lines_lst)
            for i, lines in enumerate(split_lines_lst):
                list2file(files[i], lines, add_nl=False)
        return split_lines_lst


def delete_file(file_or_dir, verbose=True):
    if os.path.exists(file_or_dir):
        if os.path.isfile(file_or_dir):  # 文件file
            os.remove(file_or_dir)
            if verbose:
                print(f'delete ok! file: {file_or_dir}')
            return True
        else:  # 目录dir
            for file_lst in os.walk(file_or_dir):
                for name in file_lst[2]:
                    os.remove(os.path.join(file_lst[0], name))
            shutil.rmtree(file_or_dir)
            if verbose:
                print(f'delete ok! dir: {file_or_dir}')
            return True
    else:
        print(f'delete false! file/dir not exists: {file_or_dir}')
        return False


def find_duplicates(in_lst):
    # duplicates = []
    # seen = set()
    # for item in in_lst:
    #     if item not in seen:
    #         seen.add(item)
    #     else:
    #         duplicates.append(item)
    # return duplicates
    c = Counter(in_lst)
    return [k for k, v in c.items() if v > 1]


def remove_duplicates(in_file, out_file=None, keep_sort=True):
    if not out_file:
        out_file = in_file
    lines = file2list(in_file)

    if keep_sort:
        out_lines = []
        tmp_set = set()
        for line in lines:
            if line not in tmp_set:
                out_lines.append(line)
                tmp_set.add(line)
    else:
        out_lines = list(set(lines))

    list2file(out_file, out_lines)


def set_items(items, keep_order=False):
    items = [tuple(item) for item in items]  # need to be hashable i.e. tuple
    ret = []
    if keep_order:
        seen = set()
        for item in items:
            if item not in seen:
                seen.append(item)
                ret.append(list(item))
        return ret
    if not keep_order:
        ret = list(map(list, set(items)))
        return ret


def sort_items(items, sort_order):
    if isinstance(sort_order, str):
        sort_order = map(int, sort_order)
    for idx in reversed(sort_order):
        items.sort(key=lambda item: item[idx])
    return items


def check_overlap(list1, list2, verbose=False):
    set1 = set(list1)
    set2 = set(list2)
    count1 = Counter(list1)
    dupli1 = [k for k, v in count1.items() if v > 1]
    count2 = Counter(list2)
    dupli2 = [k for k, v in count2.items() if v > 1]

    print(f'原始长度:{len(list1)}\t去重长度{len(set1)}\t重复项{dupli1}')
    print(f'原始长度:{len(list2)}\t去重长度{len(set2)}\t重复项{dupli2}')

    union = sorted(set1 & set2)  # 变为list
    print(f'一样的数量: {len(union)}')
    if verbose or len(union) <= 30:
        print(*union, sep='\n', end='\n\n')
    else:
        print(*union[:30], sep='\n', end=f'\n ..more(total:{len(union)})\n\n')

    a = sorted(set1 - set2)  # 变为list
    print(f'前者多了: {len(a)}')
    if verbose or len(a) <= 30:
        print(*a, sep='\n', end='\n\n')
    else:
        print(*a[:30], sep='\n', end=f'\n ..more(total:{len(a)})\n\n')

    b = sorted(set2 - set1)  # 变为list
    print(f'后者多了: {len(b)}')
    if verbose or len(b) <= 30:
        print(*b, sep='\n', end='\n\n')
    else:
        print(*b[:30], sep='\n', end=f'\n ..more(total:{len(a)})\n\n')


def print_len(files):
    if not isinstance(files, list):
        files = [files]
    len_lst = []
    for file in files:
        with open(file, 'r', encoding='U8') as f:
            len_lst.append(len(f.readlines()))
    print(len_lst, f'总和: {sum(len_lst)}')


def f(object):
    """ 格式化 """
    if 'numpy' in str(type(object)) and len(object.shape) == 1:
        object = object.tolist()
    if isinstance(object, (list, tuple)):
        if len(object) == 0:
            return ''
        if isinstance(object[0], (int, float)):
            ret = list(map(lambda e: f'{e:.2f}', object))
            return str(ret)
    return str(object)


class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def dict2obj(dct):
    if not isinstance(dct, dict):
        return dct
    inst = Dict()
    for k, v in dct.items():
        inst[k] = dict2obj(v)
    return inst


class ImmutableDict(dict):
    def __hash__(self):
        return id(self)

    def _immutable(self, *args, **kws):
        raise TypeError('object is immutable')

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    update = _immutable
    setdefault = _immutable
    pop = _immutable
    popitem = _immutable


class Any2Id():
    # fix 是否固定住字典不在增加
    def __init__(self, use_line_no=False, exist_dict=None, fix=True, counter=None):
        self.fix = fix
        self.use_line_no = use_line_no
        self.counter = Counter() if not counter else counter
        self.any2id = {}  # 内部核心dict
        if exist_dict is not None:
            self.any2id.update(exist_dict)

        # for method_name in dict.__dict__:  # 除了下面显式定义外保证能以dict的各种方法操作Any2id
        #     setattr(self, method_name, getattr(self.any2id, method_name))

    def keys(self):
        return self.any2id.keys()

    def values(self):
        return self.any2id.values()

    def items(self):
        return self.any2id.items()

    def pop(self, key):
        return self.any2id.pop(key)

    def __getitem__(self, item):
        return self.any2id.__getitem__(item)

    def __setitem__(self, key, value):
        self.any2id.__setitem__(key, value)

    def __len__(self):
        return self.any2id.__len__()

    def __iter__(self):
        return self.any2id.__iter__()

    def __str__(self):
        return self.any2id.__str__()

    def set_fix(self, fix):
        self.fix = fix

    def get(self, key, default=None, add=False):
        if not add:
            return self.any2id.get(key, default)
        else:
            # new_id = len(self.any2id)
            new_id = max(self.any2id.values()) + 1
            self.any2id[key] = new_id
            return new_id

    def get_reverse(self):
        return self.__class__(exist_dict={v: k for k, v in self.any2id.items()})

    def save(self, file, use_line_no=None, deli='\t'):
        if use_line_no is None:
            use_line_no = self.use_line_no
        items = sorted(self.any2id.items(), key=lambda e: e[1])
        out_items = [item[0] for item in items] if use_line_no else items
        list2file(out_items, file, deli=deli)
        print(f'词表文件生成成功: {file} {items[:5]}...')

    def load(self, file, use_line_no=None, deli='\t'):
        if use_line_no is None:
            use_line_no = self.use_line_no
        items = file2items(file, deli=deli)
        if use_line_no or len(items[0]) == 1:
            self.any2id = {item[0]: i for i, item in enumerate(items)}
        else:
            self.any2id = {item[0]: int(item[1]) for item in items}

    def to_count(self, any_lst):
        self.counter.update(any_lst)  # 维护一个计数器

    def reset_counter(self):
        self.counter = Counter()

    def rebuild_by_counter(self, restrict=None, min_freq=None, max_vocab_size=None):
        if not restrict:
            restrict = ['<pad>', '<unk>', '<eos>']
        freqs = self.counter.most_common()
        tokens_lst = restrict[:]
        curr_vocab_size = len(tokens_lst)
        for token, cnt in freqs:
            if min_freq and cnt < min_freq:
                break
            if max_vocab_size and curr_vocab_size >= max_vocab_size:
                break
            tokens_lst.append(token)
            curr_vocab_size += 1
        self.any2id = {token: i for i, token in enumerate(tokens_lst)}

    @classmethod
    def from_file(cls, file, use_line_no=False, deli='\t'):
        inst = cls(use_line_no=use_line_no)
        if os.path.exists(file):
            inst.load(file, deli=deli)
        else:
            # will return inst with empty any2id , e.g. boolean(inst) or len(inst) will return False
            print(f'vocab file: {file} not found, need to build and save later')
        return inst


def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.):
    if not hasattr(sequences, '__len__') or not all([hasattr(x, '__len__') for x in sequences]):
        raise ValueError(f'sequences invalid: {sequences}')

    len_lst = [len(x) for x in sequences]

    if len(set(len_lst)) == 1:  # 长度均相等
        ret = np.array(sequences)
        if maxlen is not None:
            ret = ret[:,-maxlen:] if truncating=='pre' else ret[:,:maxlen]
        return ret

    num_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(len_lst)

    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_)
    if isinstance(value, str) and dtype != object and not is_dtype_str:
        raise ValueError("`dtype` {} is not compatible with `value`'s type: {}\n"
                         "You should set `dtype=object` for variable length strings."
                         .format(dtype, type(value)))

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError(f'Truncating type not support')

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type not support')
    return x


def get_file_logger(logger_name, log_file='./qiznlp.log', level='DEBUG'):
    level = {
        'ERROR': logging.ERROR,
        'INFO': logging.INFO,
        'DEBUG': logging.DEBUG,
        'NOTSET': logging.NOTSET,
    }.get(level, logging.DEBUG)

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    fh = logging.handlers.TimedRotatingFileHandler(log_file, when="D", interval=1, backupCount=7, encoding="utf-8")
    fh.suffix = "%Y-%m-%d.log"  # 设置后缀名称，跟strftime的格式一样
    fh.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}.log$")  # _\d{2}-\d{2}
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
    logger.propagate = False  # 取消传递，只有这样才不会传到logger_root然后又在控制台也打印一遍

    return logger

def suppress_tf_warning(tf):
    import warnings
    warnings.filterwarnings('ignore')
    tf.logging.set_verbosity(tf.logging.ERROR)

if __name__ == '__main__':
    check_overlap([1, 2, 3, 4, 5, 6], [4, 5, 6, 7, 8, 9, 0])
    exit(0)

    a = Any2Id()
    data = [random.randint(1, 100) for _ in range(10000)]
    for ele in data:
        a.to_count([ele])

    a.rebuild_by_counter(['<pad>', '<unk>'])
