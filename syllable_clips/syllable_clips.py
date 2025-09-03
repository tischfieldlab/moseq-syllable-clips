from argparse import Namespace
import collections
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, wait
from typing import Dict, List, Union, cast
from typing_extensions import Literal

import h5py
import numpy as np
import pandas as pd
from moseq2_viz.viz import clean_frames
from tqdm import tqdm

from syllable_clips.session import find_session_by_id
from syllable_clips.slice import Slice, SliceInfo, expand_slice, load_h5_timestamps, prep_slice_data
from syllable_clips.util import load_timestamps
from syllable_clips.video import CropRegion, VideoReader, VideoWriter, colorize_video, crop_video, draw_onset_indicator, stack_videos




def produce_clips(syllables: List[int], examples: int, args: Namespace):
    ''' Produce `examples` number of clips for each syllable in `syllables` using `args`

    Parameters:
        syllable: (int[]) list of syllable ids to produce clips for
        examples: (int) Number of examples to produce for each syllable
        args: command line arguments
    '''
    slice_gen = prep_slice_data(args.model, args.index, sort_labels=args.sort, count=args.count)
    slice_infos: List[ExtendedSliceInfo] = []

    with ProcessPoolExecutor(max_workers=args.processors) as pool:

        for sid in tqdm(syllables, desc="Syllables"):
            slices = slice_gen(sid, args.pick)
            num_slices = len(slices)
            slice_iter = iter(slices)

            if num_slices == 0:
                tqdm.write("No slices found for syllable {}".format(sid))
                continue

            with tqdm(total=examples, desc="Examples", leave=False, disable=True) as pbar:
                all_futures = []

                def queue_item(*call_args):
                    future = pool.submit(partial_produce_clips, *call_args)
                    done_cb = make_done_callback(list(call_args))
                    all_futures.append(future)
                    future.add_done_callback(done_cb)
                #end queue_item()

                def make_done_callback(_self_args):
                    def done_callback(_f):
                        try:
                            _result = _f.result()
                            slice_infos.append(_result)
                            pbar.update(1)
                        except Exception as e:
                            logging.error(e, exc_info=True)
                            # print out the error in a nice way
                            fmt_args = [e, _self_args[1], _self_args[2] + 1]
                            tqdm.write("Something happened: {}; attempting to pick another slice for syllable {} example #{}".format(*fmt_args))

                            # requeue the work item, BUT picking another slice
                            _self_args[-1] = next(slice_iter)
                            queue_item(*_self_args)
                        finally:
                            pass
                    #end done_callback()
                    return done_callback
                #end make_done_callback()


                try:
                    for ex in range(examples):
                        queue_item(args, sid, ex, next(slice_iter))
                except StopIteration:
                    tqdm.write("Not enough slices found for syllable {}, skipping example #{}".format(sid, ex + 1))
                finally:
                    # https://stackoverflow.com/questions/38258774/python-3-how-to-properly-add-new-futures-to-a-list-while-already-waiting-upon-i
                    # wait, while allowing for possible additions of jobs to the pool
                    while all_futures:
                        fs = all_futures[:]
                        for f in fs:
                            all_futures.remove(f)
                        wait(fs)

    output_info(slice_infos, args)
#end produce_clips()


class ExtendedSliceInfo(SliceInfo):
    ''' Extended SliceInfo with additional fields for processing '''
    base_name: str
    sid_raw: int
    sid_usage: int
    sid_frames: int
    example_no: int

def partial_produce_clips(args: Namespace, sid: int, ex: int, slice: Slice) -> ExtendedSliceInfo:
    clips = None
    try:
        out_base = "{}_raw{}-usage{}-frames{}_ex{}".format(
            args.name,
            args.label_map[sid]['raw'],
            args.label_map[sid]['usage'],
            args.label_map[sid]['frames'],
            ex)

        the_slice, info = expand_slice(slice, t_prepend=args.prepend, t_append=args.append,
            manifest_path=args.manifest, manifest_session_id_col=args.man_session_id_col, manifest_uuid_col=args.man_uuid_col)
        info = cast(ExtendedSliceInfo, info)
        info['base_name'] = out_base
        info['sid_raw'] = args.label_map[sid]['raw']
        info['sid_usage'] = args.label_map[sid]['usage']
        info['sid_frames'] = args.label_map[sid]['frames']
        info['example_no'] = ex

        out_base = os.path.join(args.dir, out_base)

        clips = generate_clips(the_slice, info, args)
        output_clips(clips, out_base)
        return info
    except Exception as e:
        raise Exception("{} ({})".format(e, sys.exc_info()[2].tb_lineno))  # type: ignore[union-attr]
    finally:
        pass
        #free_clips(clips)
#end partial_produce_clips()

def generate_clips(slice: Slice, info: SliceInfo, args: Namespace) -> Dict[str, np.ndarray]:
    clips = collections.OrderedDict()
    clip = None
    try:
        for stream in args.streams:
            #print("Processing stream: {}".format(stream))
            if stream == 'rgb':
                clip = fetch_rgb_clip(slice, args.raw_path, info['session_id'], crop=args.crop_rgb, rgb_name=args.rgb_name, rgb_ts_name=args.rgb_ts_name)
                onset_size = ensure_even(int(clip.shape[2] * 0.08))
                #print("RGB Clip shape: {}".format(clip.shape))
            elif stream == 'ir':
                clip = fetch_ir_clip(slice, args.raw_path, info['session_id'], crop=args.crop_ir, ir_name=args.ir_name, ir_ts_name=args.ir_ts_name)
                onset_size = ensure_even(int(clip.shape[2] * 0.08))
            elif stream == 'depth':
                clip = fetch_depth_clip(slice, args)
                onset_size = ensure_even(int(clip.shape[2] * 0.12))
                #print("Depth Clip shape: {}".format(clip.shape))
            else:
                continue

            if args.prepend > 0 or args.append > 0:
                #print("onset size: {}".format(onset_size))
                start = info['onset_idx'] - info['start_idx']
                stop = start + (info['offset_idx'] - info['onset_idx'])
                #print(start, stop)
                clip = draw_onset_indicator(clip, start, stop, (onset_size, onset_size), color=(255, 0, 0))
                #print("Clip with onset indicator shape: {}".format(clip.shape))

            clips[stream] = clip
            clip = None
            #print(clips.keys())

        if 'composed' in args.streams:
            clips['composed'] = stack_videos(list(clips.values()), orientation='horizontal')

    except Exception as e:
        # re-raise
        raise e

    return clips
#end generate_clips()


def output_clips(clips: Dict[str, np.ndarray], basename: str) -> None:
    num_threads = 2
    for stream, clip in tqdm(clips.items(), desc="Streams", leave=False, disable=True):
        writer = VideoWriter(f'{basename}.{stream}.mp4', fps=30, threads=num_threads)
        writer.write_frames(clip)
        writer.close()
#end output_clips()

def output_info(info, args: Namespace):
    ''' info(s) '''
    if isinstance(info, (dict)):
        info = [info]
    
    df = pd.DataFrame(info)

    info_dest = os.path.join(args.dir, '{}'.format(args.name))
    cols = [
        'session_id', 'uuid', 'sid_raw', 'sid_usage', 'sid_frames', 'example_no', 'base_name', 
        'onset_idx', 'offset_idx', 'onset_time', 'offset_time', 'start_idx', 'end_idx', 'start_time', 'end_time'
    ]
    df.to_csv('{}.sources.tsv'.format(info_dest), sep='\t', index=False, columns=cols)

    with open('{}.args.json'.format(info_dest), 'w') as f:
        args_data = {k: v for k, v in vars(args).items() if k not in ['func']}
        json.dump(args_data, f, indent='\t', skipkeys=True, sort_keys=True)
#end output_info()


def fetch_depth_clip(slice: Slice, args: Namespace) -> np.ndarray:
    with h5py.File(slice[2], 'r') as h5:
        frames = clean_frames(
            h5['/frames'][slice[0][0]:slice[0][1]],
            medfilter_space=args.medfilter_space,
            gaussfilter_space=args.gaussfilter_space)

        frames = colorize_video(frames, vmin=args.min_height, vmax=args.max_height, cmap=args.cmap)

        timestamps = load_h5_timestamps(h5)[slice[0][0]:slice[0][1]]
        durations = np.diff(timestamps) / 1000

        return frames
#end fetch_depth_clip()


def fetch_rgb_clip(slice: Slice, raw_data_path: Union[str, List[str]], session_id: str, rgb_name: str = 'rgb.mp4', rgb_ts_name: str = 'rgb_ts.txt', crop: Union[Literal['none', 'auto'], dict] = 'auto') -> np.ndarray:
    with h5py.File(slice[2], 'r') as h5:
        the_session = find_session_by_id(session_id, root_dir=raw_data_path)
        if the_session is None:
            raise ValueError("Session with ID {} not found in {}".format(session_id, raw_data_path))
        rgb_path = os.path.join(the_session['directory'], rgb_name)
        rgb_timestamp_path = os.path.join(the_session['directory'], rgb_ts_name)

        rgb_times = load_timestamps(rgb_timestamp_path)
        depth_times = load_h5_timestamps(h5)
        offset = depth_times[0] - rgb_times[0]
        start = (depth_times[slice[0][0]] - depth_times[0] + offset) / 1000
        stop = (depth_times[slice[0][1]] - depth_times[0] + offset) / 1000
        slice_nframes = slice[0][1] - slice[0][0]

        reader = VideoReader(rgb_path)
        stop += 2 * (1/reader.info['fps']) # add a little extra time to ensure we get the full clip, otherwise we might miss the last frame
        clip = reader.read_frames(start, stop)

        if clip.shape[0] != slice_nframes:
            clip = clip[:slice_nframes]  # trim to expected number of frames

        if crop == 'none':
            pass
        elif crop == 'auto':
            crop_region = get_mask_bounds(h5)
            clip = crop_video(clip, crop_region)
        elif isinstance(crop, dict):
            clip = crop_video(clip, cast(CropRegion, crop))

        return clip
#end fetch_rgb_clip()


def fetch_ir_clip(slice: Slice, raw_data_path: Union[str, List[str]], session_id: str, ir_name: str = 'ir.mp4', ir_ts_name: str = 'ir_ts.txt', crop: Union[Literal['none', 'auto'], dict] = 'auto') -> np.ndarray:
    with h5py.File(slice[2], 'r') as h5:
        the_session = find_session_by_id(session_id, root_dir=raw_data_path)
        if the_session is None:
            raise ValueError("Session with ID {} not found in {}".format(session_id, raw_data_path))
        ir_path = os.path.join(the_session['directory'], ir_name)
        ir_timestamp_path = os.path.join(the_session['directory'], ir_ts_name)

        ir_times = load_timestamps(ir_timestamp_path)
        depth_times = load_h5_timestamps(h5)
        offset = depth_times[0] - ir_times[0]
        start = (depth_times[slice[0][0]] - depth_times[0] + offset) / 1000
        stop = (depth_times[slice[0][1]] - depth_times[0] + offset) / 1000
        slice_nframes = slice[0][1] - slice[0][0]

        reader = VideoReader(ir_path)
        stop += 2 * (1/reader.info['fps']) # add a little extra time to ensure we get the full clip, otherwise we might miss the last frame
        clip = reader.read_frames(start, stop)

        if clip.shape[0] != slice_nframes:
            clip = clip[:slice_nframes]  # trim to expected number of frames

        if crop == 'none':
            pass
        elif crop == 'auto':
            crop_region = get_mask_bounds(h5)
            clip = crop_video(clip, crop_region)
        elif isinstance(crop, dict):
            clip = crop_video(clip, cast(CropRegion, crop))

        return clip
#end fetch_ir_clip()


def get_mask_bounds(h5: h5py.File) -> CropRegion:
    mask = h5['/metadata/extraction/roi'][()]
    mask_idx = np.nonzero(mask)
    return {
        'x1': ensure_even(np.min(mask_idx[1])),
        'y1': ensure_even(np.min(mask_idx[0])),
        'x2': ensure_even(np.max(mask_idx[1])),
        'y2': ensure_even(np.max(mask_idx[0]))
    }
#end get_mask_bounds()

def ensure_even(num: int) -> int:
    '''Ensure that the number is even, if not, return the next higher even number.'''
    if (num % 2) == 1:
        return num + 1
    else:
        return num
#end ensure_even()
