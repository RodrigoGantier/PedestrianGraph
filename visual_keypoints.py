import os.path as osp
import glob
import cv2
import time
import numpy as np
import argparse


class FrameObjects:
    def __init__(self, frame_num, width, height):
        self.frame_num = int(frame_num) + 1
        self.ped_id = set([])
        self.width = width
        self.height = height
        self.objects = dict()

    def add_obj(self, skt, line_num):

        pnt_num = {}
        coord = np.zeros([17, 2])
        while len(skt[line_num].split(',')) > 1:

            x, y, obj_n = np.asarray(skt[line_num].split(',')).astype(np.float32)
            obj_n = int(obj_n)

            if obj_n not in self.ped_id:
                pnt_num[obj_n] = 0
                coord = np.zeros([17, 2])

            self.ped_id.add(obj_n)
            coord[pnt_num[obj_n]] = np.asarray([x, y]).astype(np.int32)
            self.objects[obj_n] = coord
            try:
                _ = skt[line_num + 1]
            except IndexError:
                break
            line_num += 1
            pnt_num[obj_n] += 1

        return line_num


def plot_fn(img, frame):

    for obj in frame.ped_id:
        for i in range(17):
            px = int(frame.objects[obj][i, 0])
            py = int(frame.objects[obj][i, 1])
            cv2.circle(img, (px, py), 2, (0, 255, 0), -1)
            fc = np.random.randint(255)
        cv2.putText(img, str(obj), (px, py), cv2.FONT_HERSHEY_SIMPLEX, 1, (fc, fc, fc), 2)

    cv2.imshow('Frame', img)
    time.sleep(0.01)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        print('stop')


def main():

    parser = argparse.ArgumentParser(description='Key-points visualization')
    parser.add_argument('-c', '--JAAD_clips', required=True, type=str, help='path to JAAD mp4 clips')
    parser.add_argument('-i', '--path_to_keypoints', required=True, type=str, help='path to key points coordinates')
    args = parser.parse_args()

    # video resolution
    width = 1280
    height = 1024

    # skt_path = './JAAD/skeleton_cpn'
    skt_path = args.path_to_keypoints

    # video_list = glob.glob('./JAAD/JAAD_clips/*.mp4')
    video_list = glob.glob(args.JAAD_clips + '/*.mp4')

    video_list.sort()
    skt_list = glob.glob(osp.join(skt_path, '*.cpn'))
    skt_list.sort()

    for i, vid_file in enumerate(video_list):

        vid_id = osp.basename(vid_file).split('.')[0]
        skt_file = osp.join(skt_path, vid_id + '.cpn')
        if skt_file not in skt_list:
            continue

        with open(skt_file, 'r') as fn:
            skt = fn.read().splitlines()
            line_num = 0
        cap = cv2.VideoCapture(vid_file)
        frame_len = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        for frame_vid in range(int(frame_len)):
            ret, img = cap.read()
            if ret:
                img = cv2.resize(img, (width, 720))
                img = cv2.copyMakeBorder(img, 152, 152, 0, 0,
                                         cv2.BORDER_CONSTANT, value=(0, 0, 0))
                frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                read_line = skt[line_num]

                # initialization frame
                if len(skt[line_num].split(',')) == 1:
                    print(read_line)
                    frame = FrameObjects(read_line, width, height)

                    if frame.frame_num == frame_num:
                        line_num += 1
                        if len(skt) == line_num:
                            break
                        line_num = frame.add_obj(skt, line_num)
                        plot_fn(img, frame)

    # release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()


