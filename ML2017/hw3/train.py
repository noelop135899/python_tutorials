from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import re
import sys
import tarfile

import urllib.request
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('data_dir', 'cifar10_data','Path to the CIFAR-10 data directory.')
flags.DEFINE_string('train_dir', 'cifar10_train',
                    'Directory where to write event logs and checkpoint.')
FLAGS = flags.FLAGS

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def download_and_extract():
    dest_directory = FLAGS.data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                                                             float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

    tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def main(_):
    download_and_extract()
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    print('done')
    """"""


if __name__ == '__main__':
    tf.app.run()