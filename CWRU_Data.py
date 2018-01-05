import numpy as np
import glob

import DTN
from sklearn.cross_validation import train_test_split


nodes_num = DTN.IMAGE_SIZE * DTN.IMAGE_SIZE

onehot_label_dict = {"normal": np.array([1, 0, 0, 0]),
              "ball": np.array([0, 1, 0, 0]),
              "inner": np.array([0, 0, 1, 0]),
              "outter": np.array([0, 0, 0, 1])}

label_dict = {"normal": np.array([1]),
              "ball": np.array([2]),
              "inner": np.array([3]),
              "outter": np.array([4])}

# x, y = extract_txt(path='data/xichu', overlap=0.8, label_dict=label_dict)
# train_X, test_X, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=0)
#
# test_X_3HP, test_y_3HP = extract_txt(path='data/xichu/3HP', overlap=0.8, label_dict=label_dict)

class Dataset(object):
    def __init__(self, images, labels):
        self._images = images
        self._labels = labels
        self._num_examples = images.shape[0]
        self._epochs_completed = 0
        self._index_in_epoch = 0


    @property
    def images(self):
        return self._images
    @property
    def labels(self):
        return self._labels
    @property
    def num_examples(self):
        return self._num_examples
    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]

def get_data(path, overlap = 0.8, label_dict = label_dict, one_hot=True):
    print('extract data from folder: %s' % path)
    txt_list = glob.glob('%s/*.txt' % path)
    x = np.empty(shape=[0, nodes_num])
    if one_hot:
        y = np.empty(shape=[0, 4])
    else:
        y = np.empty(shape=[0, 1])

    stride = nodes_num - int(overlap * nodes_num)
    for txt in txt_list:
        label = txt.split('.')[0].split('_')[len(txt.split('.')[0].split('_')) - 2]
        onehot_y = onehot_label_dict[label].reshape(1, -1)
        single_y = label_dict[label].reshape(1, -1)
        # read txt file
        or_x = np.loadtxt(txt)
        len_stamp = len(x)
        # build a slice windows to get 64*64 nodes
        for i in range(0, len(or_x) - nodes_num, stride):
            selected_x = or_x[i: i + nodes_num].reshape(1, -1)
            x = np.concatenate([x, selected_x], axis=0)
            if one_hot:
                y = np.concatenate([y, onehot_y], axis=0)
            else:
                y = np.concatenate([y, single_y], axis=0)
        print("get %s sample from %s" % (len(x) - len_stamp, txt))
    return x, y

