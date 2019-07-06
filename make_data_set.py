import os.path as osp
import os
import glob
import numpy as np
import xml.etree.cElementTree as ET
import argparse
win_size = 300


class PedestrianSkt:
    def __init__(self, vid_name, fr_len):
        self.file_name = vid_name
        self.ped_id = set({})
        self.ped_skt = dict()
        self.fr_len = fr_len

    def add_pedestrians(self, fr):

        for i in fr.ped_id:
            if i not in self.ped_id:
                assert fr.skt[i].shape == (17, 5)
                self.ped_skt[i] = fr.skt[i].reshape([1, 17, 5])
                self.ped_id.add(i)
            else:
                self.ped_skt[i] = np.vstack([self.ped_skt[i], fr.skt[i].reshape([1, 17, 5])])


class FrameSkt:
    def __init__(self, frame_num):
        self.ped_id = set([])
        self.skt = dict()
        self.frame_num = int(frame_num) + 1

    def add_skt(self, skt, line_num, fr_n, vid_num):

        pnt_num = {}
        coord = np.zeros([17, 5])

        while len(skt[line_num].split(',')) > 1:

            x, y, ped_id = np.asarray(skt[line_num].split(',')).astype(np.float32)
            ped_id = int(ped_id)

            if ped_id not in self.ped_id:
                coord = np.zeros([17, 5])
                pnt_num[ped_id] = 0

            self.ped_id.add(ped_id)
            coord[pnt_num[ped_id]] = np.asarray([x, y, fr_n, vid_num, ped_id]).astype(np.int32)
            self.skt[ped_id] = coord
            try:
                _ = skt[line_num + 1]
            except IndexError:
                break
            line_num += 1
            pnt_num[ped_id] += 1

        return line_num


def normalize_data(data, norm=False):

    all_skt  = np.zeros([0, 28])
    used_pt  = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    order_pt = [0, 2, 3, 4, 5, 6,  7,  8,  9, 10, 11, 12, 13]
    new_data = {}
    for i in range(len(data)):
        skt = data[i]
        new_skt = np.zeros([skt.shape[0], 14, 2])
        meta_data = np.zeros([skt.shape[0], 14, 3])
        skt[:, :, 0] /= 1280
        skt[:, :, 1] /= 1024

        p1 = skt[:, 5, :2]
        p2 = skt[:, 6, :2]

        x_pt = np.divide(p1[:, 0] + p2[:, 0], 2)
        y_pt = np.divide(p1[:, 1] + p2[:, 1], 2)
        m_pt = np.vstack([x_pt, y_pt]).transpose()

        new_skt[:, 1, :] = m_pt
        new_skt[:, order_pt, :] = skt[:, used_pt, :2]
        append_data = new_skt.reshape([-1, 28])

        meta_data[:, 1, :] = skt[:, 5, 2:]
        meta_data[:, order_pt, :] = skt[:, used_pt, 2:]

        new_skt = np.append(new_skt, meta_data, 2)

        new_data[i] = new_skt

        all_skt = np.append(all_skt, append_data, axis=0)

    if norm:
        skt_mean = np.mean(all_skt, axis=0)
        skt_std = np.std(all_skt, axis=0)

        for i in new_data.keys():
            skt_temp = new_data[i][:, :, :2].reshape([-1, 28])
            skt_temp = np.divide(skt_temp - skt_mean, skt_std)
            skt_temp = skt_temp.reshape([-1, 14, 2])
            new_data[i][:, :, :2] = skt_temp

    return new_data


def pre_proces(skt_path, norm=True):
    # load data----------------------------------------------

    skt_list = glob.glob(osp.join(skt_path, '*.cpn'))
    skt_list.sort()
    all_vid = []
    line_num = -1
    frame_len = -1
    for skt_file in skt_list:

        with open(skt_file, 'r') as fn:
            skt = fn.read().splitlines()
        file_name = osp.basename(skt_file).split('.')[0]
        vid = PedestrianSkt(file_name, None)

        for rd in range(len(skt) - 1, 0, -1):
            read_value = skt[rd].split(',')
            if len(read_value) == 1:
                frame_len = int(read_value[0])
                vid.fr_len = frame_len
                print('video name: {}, video len: {}'.format(file_name, frame_len + 1))
                line_num = 0
                break

        for frame_n in range(frame_len + 1):

            read_line = skt[line_num]
            data_line = read_line.split(',')
            # initialization frame
            if len(data_line) != 1:
                continue

            if frame_n == int(data_line[0]):

                frame = FrameSkt(read_line)
                line_num += 1
                if len(skt) == line_num:
                    break
                line_num = frame.add_skt(skt, line_num, frame_n, int(file_name[-4:]))
                vid.add_pedestrians(frame)
        for v_id in vid.ped_id:
            all_vid.append(vid.ped_skt[v_id])

    data_2d = normalize_data(all_vid, norm)

    # ---------------------------------------------------------------
    data_2d[217][:, :, 4] = data_2d[216][0, 0, 4]
    data_2d[218][:, :, 4] -= 1
    data_2d[219][:, :, 4] -= 1
    data_2d[216] = np.append(data_2d[216], data_2d[217], axis=0)
    del data_2d[217]

    data_2d[230][:, :, 4] = data_2d[229][0, 0, 4]
    data_2d[231][:, :, 4] -= 1
    data_2d[229] = np.append(data_2d[229], data_2d[230], axis=0)
    del data_2d[230]

    data_2d[238][:, :, 4] = data_2d[237][0, 0, 4]
    data_2d[237] = np.append(data_2d[237], data_2d[238], axis=0)
    del data_2d[238]

    data_2d[255][:, :, 4] = data_2d[254][0, 0, 4]
    data_2d[254] = np.append(data_2d[254], data_2d[255], axis=0)
    data_2d[256][:, :, 4] -= 1
    data_2d[257][:, :, 4] -= 2
    data_2d[258][:, :, 4] -= 2
    data_2d[256] = np.append(data_2d[256], data_2d[257], axis=0)
    del data_2d[255]
    del data_2d[257]

    data_2d[260][:, :, 4] = data_2d[259][0, 0, 4]
    data_2d[259] = np.append(data_2d[259], data_2d[260], axis=0)
    data_2d[261][:, :, 4] -= 1
    data_2d[262][:, :, 4] -= 1
    del data_2d[260]

    data_2d[273][:, :, 4] = data_2d[272][0, 0, 4]
    data_2d[272] = np.append(data_2d[272], data_2d[273], axis=0)
    del data_2d[273]

    data_2d[369][:, :, 4] = data_2d[368][0, 0, 4]
    data_2d[368] = np.append(data_2d[368], data_2d[369], axis=0)
    del data_2d[369]

    data_2d[480][:, :, 4] = data_2d[479][0, 0, 4]
    data_2d[479] = np.append(data_2d[479], data_2d[480], axis=0)
    del data_2d[480]
    data_2d[481][:, :, 4] -= 1
    data_2d[482][:, :, 4] -= 1

    data_2d[559][:, :, 4] = data_2d[558][0, 0, 4]
    data_2d[558] = np.append(data_2d[558], data_2d[559], axis=0)
    del data_2d[559]
    data_2d[560][:, :, 4] -= 1
    data_2d[561][:, :, 4] -= 1
    data_2d[562][:, :, 4] -= 1
    data_2d[563][:, :, 4] -= 1

    data_2d[673][:, :, 4] = data_2d[672][0, 0, 4]
    data_2d[672] = np.append(data_2d[672], data_2d[673], axis=0)
    del data_2d[673]
    data_2d[674][:, :, 4] -= 1
    data_2d[675][:, :, 4] -= 1

    new_data_2d = {}
    for num_, keys_ in enumerate(data_2d.keys()):
        new_data_2d[num_] = data_2d[keys_]
    # ---------------------------------------------------------------
    return new_data_2d


def shuffle_backward(l, order):
    l_out = [0] * len(l)
    for i, j in enumerate(order):
        l_out[j] = l[i]
    return np.asarray(l_out)


def save_file_3d(path_save, save_data):

    data_len = save_data.shape[0]
    with open(path_save, 'w+') as file:
        for frame_num in range(data_len):
            file.write('{}\n'.format(frame_num))
            for body_part in range(save_data.shape[1]):
                file.write('{},{},{}\n'.format(
                    save_data[frame_num, body_part, 0],
                    save_data[frame_num, body_part, 1],
                    save_data[frame_num, body_part, 2]
                ))


def save_file_2d(path_save, save_data):

    data_len = save_data.shape[0]
    with open(path_save, 'w+') as file:
        for frame_num in range(data_len):
            file.write('{}\n'.format(frame_num))
            for body_part in range(save_data.shape[1]):
                file.write('{},{}\n'.format(
                    save_data[frame_num, body_part, 0],
                    save_data[frame_num, body_part, 1]
                ))


def save_file_data(path_save, save_data):

    if save_data.shape[0] <= win_size:
        save_data = np.pad(save_data, [(0, win_size - save_data.shape[0]), (0, 0), (0, 0)], mode='constant')
        save_file_2d(path_save, save_data)
    else:
        list_data = [*range(save_data.shape[0])]
        it = iter(list_data)
        win = []
        for e in range(win_size):
            win.append(next(it))
        new_path = path_save[:-4] + '_w000.cpn'

        save_file_2d(new_path, save_data[win])
        cont = 1
        for e in it:
            win = win[1:] + [e]
            new_path = path_save[:-4] + '_w{:03d}.cpn'.format(cont)

            save_file_2d(new_path, save_data[win])
            cont += 1


def ped_prosses(ped_num, ped_list):
    if ped_num == 0:
        ped_idx = ped_list.index('pedestrian')
    else:
        ped_idx = ped_list.index('pedestrian{}'.format(ped_num))
    return ped_idx


def ped_prosses2(ped_num, ped_list):

    if ped_num == 0:
        try:
            ped_idx = ped_list.index('pedestrian')
        except ValueError:
            ped_idx = ped_list.index('pedestrian1')
    else:
        ped_idx = ped_list.index('pedestrian{}'.format(ped_num + 1))

    return ped_idx


def save_file(pred_batch, paths_dir, metadata, xml_path):

    pred_batch = pred_batch.reshape([pred_batch.shape[0], 14, 2])
    metadata = np.asarray(metadata, dtype=np.int32)

    xml_file = osp.join(xml_path, 'video_{:04d}.xml'.format(metadata[0, 0, 1]))
    xml_root = ET.parse(xml_file).getroot()

    print_data = 'video_{:04d}_ped{:04d}'.format(metadata[0, 0, 1], metadata[0, 0, 2])
    print(print_data)

    ped_list = [i.tag for i in xml_root[2][1:]]
    ped_num = metadata[0, 0, 2]

    if 'pedestrian' in ped_list and 'pedestrian1'in ped_list:
        ped_idx = ped_prosses(ped_num, ped_list)
    else:
        ped_idx = ped_prosses2(ped_num, ped_list)

    actions = xml_root[2][ped_idx + 1]

    crossing = False
    p1 = 0
    for act in actions:
        if act.get('id') == 'crossing' or act.get('id') == 'CROSSING':
            end_frame = int(act.get('end_frame')) - 1
            start_frame = int(act.get('start_frame')) - 1
            crossing = True

    if crossing:
        p1 = int(np.where(metadata[:, 0, 0] == start_frame)[0].mean())
        p2 = int(np.where(metadata[:, 0, 0] == end_frame)[0].mean())
        save_data = pred_batch[p1:p2]  # do not need

    # if p1 > win_size and crossing:
    if crossing and p1 > 1:
        save_data = pred_batch[:p1]
        vid_len = int(win_size*1.5)
        save_data = save_data[-vid_len:]  # !!!!!!!!!!!!!!!!!!!!

        base_name = 'Video_{:04d}_ped_{:04d}'.format(metadata[0, 0, 1], metadata[0, 0, 2])

        path_save = osp.join(paths_dir, 'will_cro2d', base_name + '.cpn')
        save_file_data(path_save, save_data)

    if not crossing:
        save_data = pred_batch

        base_name = 'Video_{:04d}_ped_{:04d}'.format(metadata[0, 0, 1], metadata[0, 0, 2])
        path_save = osp.join(paths_dir, 'no_cro2d', base_name + '.cpn')
        save_file_data(path_save, save_data)


def train(jaad_2d, save_path, xml_path):

    numb_batches = len(jaad_2d)
    for bat_n in range(numb_batches):
        batch_2d = jaad_2d[bat_n][:, :, :2]
        batch_2d = batch_2d.reshape([batch_2d.shape[0], 28])

        meta_data = jaad_2d[bat_n][:, :, 2:]
        save_file(batch_2d, save_path, meta_data, xml_path)


def main():

    parser = argparse.ArgumentParser(description='Organize data')
    parser.add_argument('-c', '--skeleton_path', required=True, type=str, help='CPN skeleton path')
    parser.add_argument('-i', '--xml_labels', required=True, type=str, help='path to xml labels')
    parser.add_argument('-n', '--save_path', required=True, type=str, help='path to save organized data')

    args = parser.parse_args()

    # skeleton_path = '/media/rodrigo/data/JAAD/skeleton_cpn'
    skeleton_path = args.skeleton_path
    # save_txt_path = '/media/rodrigo/data/JAAD/output_2d'
    save_txt_path = args.save_path
    # xml_path = '/media/rodrigo/data/JAAD/JAAD_behavior-master/behavioral_data_xml'
    xml_path = args.xml_labels

    if not osp.isdir(osp.join(save_txt_path, 'will_cro2d')):
        os.mkdir(osp.join(save_txt_path, 'will_cro2d'))
    if not osp.isdir(osp.join(save_txt_path, 'no_cro2d')):
        os.mkdir(osp.join(save_txt_path, 'no_cro2d'))

    norm = False
    data_jaad = pre_proces(skeleton_path, norm=norm)

    train(data_jaad, save_txt_path, xml_path)

    print('finish')


if __name__ == '__main__':
    main()

