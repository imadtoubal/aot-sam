import gc
import glob
import os
import sys
import time

import cv2
import numpy as np
import torch
from vot.region import Mask
from vot.region import io as vot_io
from vot.region.io import read_trajectory

from helpers import draw_mask, save_prediction
from model_args import aot_args, sam_args, segtracker_args
from SegTracker import SegTracker


def main():
    DATASET = '/usr/mvl2/itdfh/dev/vot-mixformer-sam/sequences'
    # DATASET = '/usr/mvl2/itdfh/dev/vot-development/sequences'
    # SEQ = 'cat'
    seq_name = sys.argv[1]
    seq = os.path.join(DATASET, seq_name)

    imgs_paths = sorted(glob.glob(os.path.join(seq, 'color/*.jpg')))
    gt_paths = sorted(glob.glob(os.path.join(seq, 'groundtruth*.txt')))
    print('gt_paths', gt_paths)
    first_img = cv2.imread(imgs_paths[0])
    initial_mask = np.zeros(first_img.shape[:2], dtype=np.uint8)
    tracks = {}
    for i, gt_path in enumerate(gt_paths, 1):
        curr_obj = read_trajectory(gt_path)[0]
        bounds = curr_obj.bounds()
        x1, y1, x2, y2 = bounds
        initial_mask[y1:y2+1, x1:x2+1] = curr_obj.mask * i
        tracks[i] = []

    seg_tracker = SegTracker(segtracker_args, sam_args, aot_args)
    seg_tracker.restart_tracker()

    io_args = {
      'output_video': f'output_videos/{seq_name}.mp4',
      'output_masked_frame_dir': f'masked_frames/{seq_name}',
      'output_vot_dir': f'output_vot/{seq_name}',
    }

    # create dir to save predicted mask and masked frame
    output_mask_dir = f'output/{seq_name}'
    os.makedirs(output_mask_dir, exist_ok=True)
    os.makedirs(io_args['output_masked_frame_dir'], exist_ok=True)
    os.makedirs(io_args['output_vot_dir'], exist_ok=True)

    pred_list = []
    masked_pred_list = []

    torch.cuda.empty_cache()
    gc.collect()
    frame_idx = 0

    with torch.cuda.amp.autocast():
        times = []
        for img_path in imgs_paths:
            start_time = time.time()
            frame_name = os.path.basename(img_path).split('.')[0]
            frame = cv2.imread(img_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if frame_idx == 0:
                seg_tracker.add_reference(frame, initial_mask)
                torch.cuda.empty_cache()
                gc.collect()
                pred_mask = initial_mask
            else:
                pred_mask = seg_tracker.track(frame, update_memory=True)
            torch.cuda.empty_cache()
            gc.collect()

            for obj_idx in tracks:
                obj_mask = (pred_mask == obj_idx).astype(np.uint8)
                vot_mask = Mask(obj_mask)
                tracks[obj_idx].append(vot_mask)

            end_time = time.time()
            times.append(end_time - start_time)
            save_prediction(pred_mask, output_mask_dir, f'{frame_name}.png')
            pred_list.append(pred_mask)
            print("processed frame {}, obj_num {}".format(
                frame_idx, seg_tracker.get_obj_num()), end='\r')
            frame_idx += 1
        print('\nfinished')

    ##################
    # Save tracks
    ##################
    for obj_idx in tracks:
        ofile = os.path.join(io_args['output_vot_dir'],
                             f'{seq_name}_{obj_idx}_001.bin')
        ofile_time = os.path.join(io_args['output_vot_dir'],
                                  f'{seq_name}_{obj_idx}_001_time.value')
        with open(ofile, 'wb') as f:
            vot_io.write_trajectory_binary(f, tracks[obj_idx])

        with open(ofile_time, 'w') as f:
            for t in times:
                f.write(f'{t/len(tracks)}\n')

    ##################
    # Visualization
    ##################

    # draw pred mask on frame and save as a video
    height, width = pred_list[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    fps = 30
    out = cv2.VideoWriter(
        io_args['output_video'], fourcc, fps, (width, height))

    frame_idx = 0
    for img_path in imgs_paths:
        frame_name = os.path.basename(img_path).split('.')[0]
        frame = cv2.imread(img_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pred_mask = pred_list[frame_idx]
        masked_frame = draw_mask(frame, pred_mask)
        masked_pred_list.append(masked_frame)
        cv2.imwrite(f"{io_args['output_masked_frame_dir']}/{frame_name}.png",
                    masked_frame[:, :, ::-1])

        masked_frame = cv2.cvtColor(masked_frame, cv2.COLOR_RGB2BGR)
        out.write(masked_frame)
        print('frame {} writed'.format(frame_name), end='\r')
        frame_idx += 1
    out.release()
    print("\n{} saved".format(io_args['output_video']))
    print('\nfinished')

    torch.cuda.empty_cache()
    gc.collect()


if __name__ == '__main__':
    main()
