from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse
import csv
import pandas as pd
import collections
import cv2
import sys
import torch
from itertools import count
import numpy as np
from glob import glob
import json
import time
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker


torch.set_num_threads(1)

# parser = argparse.ArgumentParser(description='tracking demo')
# parser.add_argument('--video_name', default='', type=str, help='videos or image files')
# args = parser.parse_args()


def get_frames(video_name):

    if video_name.endswith('avi') or \
        video_name.endswith('mp4'):
        cap = cv2.VideoCapture(video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = glob(os.path.join(video_name, '*.jp*'))
        images = sorted(images,
                        key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame
    
# def main():

#     # load config
#     cfg.merge_from_file('./experiments/siamrpn_alex_dwxcorr/config.yaml')
#     cfg.CUDA = torch.cuda.is_available()
#     device = torch.device('cuda' if cfg.CUDA else 'cpu')

#     # create model
#     model = ModelBuilder()

#     # load model
#     model.load_state_dict(torch.load('./experiments/siamrpn_alex_dwxcorr/model.pth', map_location=lambda storage, loc: storage.cpu()))
#     model.eval().to(device)

#     # build tracker
#     tracker = build_tracker(model)

#     first_frame = True
#     if args.video_name:
#         video_name = args.video_name.split('/')[-1].split('.')[0]
#     else:
#         exit()

#     for frame in get_frames(args.video_name):
#         if first_frame:
#             try:
#                 #init_rect = cv2.selectROI(video_name, frame, False, False)
#                 # with open('./demo/groundtruth.csv', 'r') as f:
#                 #    reader = csv.reader(f)
#                 #    init_rect = list(reader)
#                 # df = pd.read_csv('./demo/groundtruth.csv', delimiter=',')
#                 # init_rect = [tuple(x) for x in df.values]

#                 init_rect = [252, 103, 157, 187]
#             except:
#                 exit()
#             tracker.init(frame, init_rect)
#             first_frame = False
#         else:
#             outputs = tracker.track(frame)

#             if 'polygon' in outputs:
#                 polygon = np.array(outputs['polygon']).astype(np.int32)
#                 cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
#                               True, (0, 255, 0), 3)
#                 mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
#                 mask = mask.astype(np.uint8)
#                 mask = np.stack([mask, mask*255, mask]).transpose(1, 2, 0)
#                 frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
#             else:
#                 bbox = list(map(int, outputs['bbox']))
#                 cv2.rectangle(frame, (bbox[0], bbox[1]),
#                               (bbox[0]+bbox[2], bbox[1]+bbox[3]),
#                               (0, 255, 0), 3)

#                 for frame in get_frames(args.video_name):   
#                     mylist = [video_name, bbox]

#                 with open(r'coordinates.csv', 'a', newline='') as csvfile:
#                     writer = csv.writer(csvfile)
#                     writer.writerow(mylist)

#             with open('coordinates.csv') as inp, open('frame-tracking-cords.csv', 'w') as out:
#                 reader = csv.reader(inp)
#                 writer = csv.writer(out, delimiter=',')
#                 writer.writerow(['0'] + next(reader))
#                 writer.writerows([i] + row for i, row in enumerate(reader, 1))

# if __name__ == '__main__':
#     main()