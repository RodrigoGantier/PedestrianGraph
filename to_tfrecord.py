import numpy as np
import tensorflow as tf
import sys
import os.path as osp
import os
import glob
import random
import argparse


class data_stats:

    def __init__(self):
        self.video_num = []
        self.ped_id = {}
        self.win_num = []
        self.path_data = {}

    def add_item(self, item):

        base_name = osp.basename(item)
        if len(base_name) > 23:  # this sample has more than 1 window
            vid_num = int(base_name[6:10])
            ped_num = int(base_name[15:19])
            win_num = int(base_name[21:24])

            ped_id = '{:04d}_{:04d}'.format(vid_num, ped_num)
            if ped_id not in self.ped_id:
                self.ped_id[ped_id] = [win_num]
                self.path_data[ped_id] = [item]
            else:
                self.ped_id[ped_id].append(win_num)
                self.path_data[ped_id].append(item)

            if vid_num not in self.video_num:
                self.video_num.append(vid_num)

            self.win_num.append(win_num)
        else:
            vid_num = int(base_name[6:10])
            ped_num = int(base_name[15:19])

            ped_id = '{:04d}_{:04d}'.format(vid_num, ped_num)
            if ped_id not in self.ped_id:
                self.ped_id[ped_id] = [0]
                self.path_data[ped_id] = [item]
            else:
                self.ped_id[ped_id].append(0)
                self.path_data[ped_id] = [item]

            if vid_num not in self.video_num:
                self.video_num.append(vid_num)

            self.win_num.append(0)


def data_sample(data):

    tr_list = []
    te_list = []

    for data_key in data.ped_id.keys():
        data_path = data.path_data[data_key]
        samp_len = min(3, len(data_path))
        for i in range(samp_len):
            # choose firth 250 samples for training and the rest for test
            if int(osp.basename(data_path[-i])[6:10]) < 251:
                tr_list.append(data_path[-i])
            else:
                te_list.append(data_path[-i])

    random.shuffle(tr_list)
    random.shuffle(te_list)
    return tr_list, te_list


def load_data(file_path):

    # gather paths.
    no_cross = osp.join(file_path, 'no_cro2d/*.cpn')
    will_cross = osp.join(file_path, 'will_cro2d/*.cpn')

    # gather all files within the paths and shuffle.
    no_crs_data = glob.glob(no_cross)
    no_crs_data.sort()
    will_crs_data = glob.glob(will_cross)
    will_crs_data.sort()

    # initialize data class.
    no_crs_stat = data_stats()
    will_crs_stat = data_stats()

    # add samples to each class.
    for item in no_crs_data:
        no_crs_stat.add_item(item)
    for item in will_crs_data:
        will_crs_stat.add_item(item)

    # choose first 250 classes for train and the rest for test.
    no_crs_tr, no_crs_te = data_sample(no_crs_stat)
    will_crs_tr, will_crs_te = data_sample(will_crs_stat)

    # valance data-set.
    trim_tr = min(len(no_crs_tr), len(will_crs_tr))
    trim_te = min(len(no_crs_te), len(will_crs_te))

    tr_data = no_crs_tr[:trim_tr] + will_crs_tr[:trim_tr]
    te_data = no_crs_te[:trim_te] + will_crs_te[:trim_te]

    # initialize lists.
    tr_label = [0] * trim_tr + [1] * trim_tr
    te_label = [0] * trim_te + [1] * trim_te

    # initialize shuffle ids.
    tr_shuff_list = random.sample([*range(len(tr_data))], len(tr_data))
    te_shuff_list = random.sample([*range(len(te_data))], len(te_data))

    # shuffle with ids list.
    tr_data = [tr_data[i] for i in tr_shuff_list]
    te_data = [te_data[i] for i in te_shuff_list]

    tr_label = [tr_label[i] for i in tr_shuff_list]
    te_label = [te_label[i] for i in te_shuff_list]

    return tr_data, te_data, tr_label, te_label


def wrap_int64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def wrap_float(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def wrap_bytes(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def print_progress(count, total):
    # Percentage completion.
    pct_complete = float(count) / total

    # Status-message.
    # Note the \r which means the line should overwrite itself.
    msg = "\r- Progress: {0:.1%}".format(pct_complete)

    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()


def read_skeleton(path_file, label_file):

    time_size = 300  # time window.
    num_points = 14  # number of key-points.
    # initialize vector [time window, num key-points, dim=2D].
    key_point_2d = np.zeros([time_size, num_points, 2])

    with open(path_file) as file:
        df_cpn = file.read().splitlines()

    # iterate across time window.
    for frame_num in range(time_size):
        pos = ((num_points + 1) * frame_num) + 1
        coord_kps = df_cpn[pos:pos + num_points]

        # iterate across each key point.
        for num_, coord_kp in enumerate(coord_kps):

            coord_x_2d = float(coord_kp.split(",")[0])
            coord_y_2d = float(coord_kp.split(",")[1])
            # copy each key-point in its corresponding place.
            key_point_2d[frame_num, num_, :] = coord_x_2d, coord_y_2d

    return key_point_2d.reshape(-1), label_file


def convert(skt_paths, skt_labels, out_path):

    print("Converting: " + out_path)

    # Number of images. Used when printing the progress.
    num_skt = len(skt_paths)

    # Open a TFRecordWriter for the output-file.
    with tf.python_io.TFRecordWriter(out_path) as writer:
        # Iterate over all the image-paths and class-labels.
        for i, path_label in enumerate(zip(skt_paths, skt_labels)):
            # Print the percentage-progress.
            print_progress(count=i, total=num_skt - 1)

            # Load the txt-file.
            path, label = path_label
            key_points, class_vec = read_skeleton(path, label)

            # Create a dict with the data we want to save in the
            # TFRecords file. You can add more relevant data here.
            data = {
                    'image': wrap_float(key_points),
                    # 'image': wrap_bytes(key_points),
                    'label': wrap_int64(class_vec)
                    }

            # Wrap the data as TensorFlow Features.
            feature = tf.train.Features(feature=data)

            # Wrap again as a TensorFlow Example.
            example = tf.train.Example(features=feature)

            # Serialize the data.
            serialized = example.SerializeToString()

            # Write the serialized data to the TFRecords file.
            writer.write(serialized)


def main():

    parser = argparse.ArgumentParser(description='Buil tfrecod for train and test')
    parser.add_argument('-c', '--path_to_file', required=True, type=str, help='data path')
    args = parser.parse_args()

    save_path = osp.join(args.path_to_file, 'tfrecords')
    # if no save directory, create one
    if not osp.isdir(save_path):
        os.mkdir(save_path)

    # load data-set
    tr_data, te_data, tr_label, te_label = load_data(args.path_to_file)

    convert(tr_data, tr_label, osp.join(save_path, 'train_2d_pick.tfrecords'))
    convert(te_data, te_label, osp.join(save_path, 'test_2d_pick.tfrecords'))
    print('Finish')


if __name__ == '__main__':
    main()


