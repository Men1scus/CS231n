import os, json
import numpy as np
import h5py

dir_path = os.path.dirname(os.path.realpath(__file__))
BASE_DIR = os.path.join(dir_path, "datasets/coco_captioning")

def load_coco_data(base_dir=BASE_DIR, max_train=None, pca_features=True):
    print('base dir ', base_dir)
    data = {}
    caption_file = os.path.join(base_dir, "coco2014_captions.h5")
    with h5py.File(caption_file, "r") as f: # 只读方式打开
        for k, v in f.items():
            data[k] = np.asarray(v) # 每项储存成 numpy 数组的格式

    if pca_features:
        train_feat_file = os.path.join(base_dir, "train2014_vgg16_fc7_pca.h5")
    else:
        train_feat_file = os.path.join(base_dir, "train2014_vgg16_fc7.h5")
    with h5py.File(train_feat_file, "r") as f:
        data["train_features"] = np.asarray(f["features"])

    if pca_features:
        val_feat_file = os.path.join(base_dir, "val2014_vgg16_fc7_pca.h5")
    else:
        val_feat_file = os.path.join(base_dir, "val2014_vgg16_fc7.h5")
    with h5py.File(val_feat_file, "r") as f:
        data["val_features"] = np.asarray(f["features"])

    dict_file = os.path.join(base_dir, "coco2014_vocab.json") # 词汇表文件
    with open(dict_file, "r") as f:
        dict_data = json.load(f)
        for k, v in dict_data.items():
            data[k] = v

    train_url_file = os.path.join(base_dir, "train2014_urls.txt")
    with open(train_url_file, "r") as f:
        train_urls = np.asarray([line.strip() for line in f]) # 去除每行的首尾空白
    data["train_urls"] = train_urls

    val_url_file = os.path.join(base_dir, "val2014_urls.txt")
    with open(val_url_file, "r") as f:
        val_urls = np.asarray([line.strip() for line in f])
    data["val_urls"] = val_urls

    # Maybe subsample the training data
    if max_train is not None:
        num_train = data["train_captions"].shape[0]
        mask = np.random.randint(num_train, size=max_train)
        data["train_captions"] = data["train_captions"][mask]
        data["train_image_idxs"] = data["train_image_idxs"][mask]
#         data["train_features"] = data["train_features"][data["train_image_idxs"]]
    return data


def decode_captions(captions, idx_to_word):
    singleton = False
    if captions.ndim == 1:
        singleton = True # 只有一个标题
        captions = captions[None] # 增加一个维度，将大小为 N 的一维数组转化为 1xN 的二维数组
    decoded = []
    N, T = captions.shape # N 个caption，最大长度为T，中间用<NULL>来分割，最后用<END>来终止
    for i in range(N):
        words = [] # 用来保存每行 caption
        for t in range(T):
            word = idx_to_word[captions[i, t]]
            if word != "<NULL>":
                words.append(word)
            if word == "<END>":
                break
        decoded.append(" ".join(words)) # 所有行的 caption 都保存在 decoded 中
    if singleton:
        decoded = decoded[0] # 转化为一个字符串而不是多个字符串的列表
    return decoded


def sample_coco_minibatch(data, batch_size=100, split="train"):
    split_size = data["%s_captions" % split].shape[0] # caption的总数
    mask = np.random.choice(split_size, batch_size) # 抽取 batch_size 个下标 
    captions = data["%s_captions" % split][mask]
    image_idxs = data["%s_image_idxs" % split][mask]
    image_features = data["%s_features" % split][image_idxs]
    urls = data["%s_urls" % split][image_idxs]
    return captions, image_features, urls
