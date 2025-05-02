#!/usr/bin/env python

import argparse
import collections
import json
import os
import shutil
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor, wait

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from moseq2_viz.viz import clean_frames
from moviepy.editor import (CompositeVideoClip, ImageClip, ImageSequenceClip,
                            VideoFileClip, clips_array)
from moviepy.video.fx.crop import crop as moviepy_crop
from tqdm import tqdm

from syllable_clips.slice import expand_slice, load_h5_timestamps, prep_slice_data
from moseq_session_io.inspect import find_sessions_path, load_timestamps





def produce_clips(syllables, examples, args):
    ''' Produce `examples` number of clips for each syllable in `syllables` using `args`

    Parameters:
        syllable: (int[]) list of syllable ids to produce clips for
        examples: (int) Number of examples to produce for each syllable
        args: command line arguments
    '''
    slice_gen = prep_slice_data(args.model, args.index, sort_labels=args.sort, count=args.count)
    slice_infos = []

    with ProcessPoolExecutor(max_workers=args.processors) as pool:

        for sid in tqdm(syllables, desc="Syllables"):
            slices = slice_gen(sid, args.pick)
            num_slices = len(slices)
            slices = iter(slices)

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
                            # print out the error in a nice way
                            fmt_args = [e, _self_args[1], _self_args[2] + 1]
                            tqdm.write("Something happened: {}; attempting to pick another slice for syllable {} example #{}".format(*fmt_args))

                            # requeue the work item, BUT picking another slice
                            _self_args[-1] = next(slices)
                            queue_item(*_self_args)
                        finally:
                            pass
                    #end done_callback()
                    return done_callback
                #end make_done_callback()


                try:
                    for ex in range(examples):
                        queue_item(args, sid, ex, next(slices))
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

def partial_produce_clips(args, sid, ex, slice):
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
        raise Exception("{} ({})".format(e, sys.exc_info()[2].tb_lineno))
    finally:
        free_clips(clips)
#end partial_produce_clips()

def generate_clips(slice, info, args):
    clips = collections.OrderedDict()
    clip = None
    try:
        for stream in args.streams:
            if stream == 'rgb':
                clip = fetch_rgb_clip(slice, args.raw_path, info['session_id'], scratch_dir=args.scratch, crop=args.crop_rgb)
                onset_size = ensure_even(int(clip.w * 0.08))
                onset = fetch_onset_clip((onset_size, onset_size))
            if stream == 'ir':
                clip = fetch_ir_clip(slice, args.raw_path, info['session_id'], scratch_dir=args.scratch, crop=args.crop_ir)
                onset_size = ensure_even(int(clip.w * 0.08))
                onset = fetch_onset_clip((onset_size, onset_size))
            elif stream == 'depth':
                clip = fetch_depth_clip(slice, args)
                onset_size = ensure_even(int(clip.w * 0.12))
                onset = fetch_onset_clip((onset_size, onset_size))
            else:
                continue

            if args.prepend > 0 or args.append > 0:
                clip = CompositeVideoClip([
                    clip,
                    onset.set_start(args.prepend).set_duration(slice[3])
                ]).set_duration(clip.duration)

            clips[stream] = clip
            clip = None

        if 'composed' in args.streams:
            clips['composed'] = compose_clips(list(clips.values()))

    except Exception as e:
        # close any clips
        if clip is not None:
            clip.close()
        free_clips(clips)
        # re-raise
        raise e

    return clips
#end generate_clips()

def free_clips(clips):
    if clips is not None:
        for clip in clips.values():
            clip.close()
#end free_clips()

def output_clips(clips, basename):
    num_threads = 2
    for stream, clip in tqdm(clips.items(), desc="Streams", leave=False, disable=True):
        clip.write_videofile('{}.{}.mp4'.format(basename, stream), fps=30, audio=False, logger=None, threads=num_threads)
#end output_clips()

def output_info(info, args):
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

def compose_clips(clips):
    """ if len(clips) == 0 or not isinstance(clips, list):
        raise RuntimeError('Need at least one clip to compose!')
    if len(clips) == 1:
        return clips[0] """

    return clips_array([clips])
#compose_clips()

def fetch_onset_clip(size=(20,20)):
    half_x = size[0] / 2
    half_y = size[1] / 2
    r = min(half_x, half_y)

    xx, yy = np.mgrid[:size[0], :size[1]]
    circle_dist = (xx - half_x) ** 2 + (yy - half_y) ** 2
    c = np.zeros((size[0],size[1],3), dtype=np.uint8)
    cm = np.zeros((size[0],size[1]))
    c[circle_dist < r**2] = [255,0,0]
    cm[circle_dist < r**2] = 1

    clip = ImageClip(c).set_mask(ImageClip(cm, ismask=True))
    return clip
#end fetch_onset_clip()

def fetch_depth_clip(slice, args):
    with h5py.File(slice[2], 'r') as h5:
        frames = clean_frames(
            h5['/frames'][slice[0][0]:slice[0][1]],
            medfilter_space=args.medfilter_space,
            gaussfilter_space=args.gaussfilter_space)
        #moviepy wants a python LIST of numpy arrays not a numpy ARRAY of numpy arrays
        images = []
        use_cmap = plt.get_cmap(args.cmap)
        for frame in frames:
            img = frame.copy().astype('float32')
            img = (img - args.min_height) / (args.max_height - args.min_height)
            img[img < 0] = 0
            img[img > 1] = 1
            img = np.delete(use_cmap(img), 3, 2)*255
            images.append(img)

        timestamps = load_h5_timestamps(h5)[slice[0][0]:slice[0][1]]
        durations = np.diff(timestamps) / 1000

        clip = ImageSequenceClip(images, durations=list(durations)).size
        return clip
#end fetch_depth_clip()


def fetch_rgb_clip(slice, raw_data_path, session_id, scratch_dir='', rgb_name='rgb.mp4', rgb_ts_name='rgb_ts.txt', crop='auto'):
    with h5py.File(slice[2], 'r') as h5:
        sessions = find_sessions_path(raw_data_path)
        matching_sessions = list(filter(lambda s: s['session_id'] == session_id, sessions))
        if len(matching_sessions) <= 0:
            raise FileNotFoundError("Could not find RGB video for {} on path {}".format(session_id, raw_data_path))
        else:
            the_session = matching_sessions[0]

        if the_session['is_compressed']:
            scratch_dir = os.path.join(scratch_dir, session_id)

            # check if this was previously extracted by us.
            if not os.path.exists(os.path.join(scratch_dir, rgb_name)):
                session_path = os.path.join(the_session['directory'], the_session['session_id']+'.tar.gz')

                with ProgressFileObject(session_path) as pfo:
                    pfo.progress.disable = True
                    pfo.progress.set_description('Scanning tarball {}'.format(the_session['session_id']))
                    pfo.progress.leave = False
                    try:
                        tar = tarfile.open(fileobj=pfo, mode='r:gz')
                        tar_members = tar.getmembers()
                        tar_names = [_.name for _ in tar_members]
                        to_extract = [
                            tar_members[tar_names.index(rgb_name)],
                            tar_members[tar_names.index(rgb_ts_name)]
                        ]
                        ensure_dir(scratch_dir)
                        pfo.progress.set_description('Extracting')
                        pfo.progress.reset(total=sum([m.size for m in to_extract]))
                        tar.extractall(path=scratch_dir, members=to_extract)
                    except tarfile.TarError as e:
                        raise type(e)("{}: {}".format(the_session['session_id'], str(e))).with_traceback(sys.exc_info()[2])
            else:
                pass #sys.stderr.write('Found previously extracted; using that...\n')

            rgb_path = os.path.join(scratch_dir, rgb_name)
            rgb_timestamp_path = os.path.join(scratch_dir, rgb_ts_name)
        else:
            rgb_path = os.path.join(the_session['directory'], rgb_name)
            rgb_timestamp_path = os.path.join(the_session['directory'], rgb_ts_name)

        clip = VideoFileClip(rgb_path, fps_source="tbr")

        # crop time to the specified slice (synching with depth data)
        rgb_times = load_timestamps(rgb_timestamp_path)
        depth_times = load_h5_timestamps(h5)[slice[0][0]:slice[0][1]]
        rgb_time_start = (nearest(rgb_times, depth_times[0]) - rgb_times[0]) / 1000
        rgb_time_end = (nearest(rgb_times, depth_times[-1]) - rgb_times[0]) / 1000
        clip = clip.subclip(rgb_time_start, rgb_time_end)

        if crop == 'none':
            pass
        elif crop == 'auto':
            crop_region = get_mask_bounds(h5)
            clip = moviepy_crop(clip, **crop_region)
        elif isinstance(crop, dict):
            clip = moviepy_crop(clip, **crop)

        return clip
#end fetch_rgb_clip()


def fetch_ir_clip(slice, raw_data_path, session_id, scratch_dir='', ir_name='ir.mp4', ir_ts_name='ir_ts.txt', crop='auto'):
    with h5py.File(slice[2], 'r') as h5:
        sessions = find_sessions_path(raw_data_path)
        matching_sessions = list(filter(lambda s: s['session_id'] == session_id, sessions))
        if len(matching_sessions) <= 0:
            raise FileNotFoundError("Could not find IR video for {} on path {}".format(session_id, raw_data_path))
        else:
            the_session = matching_sessions[0]

        if the_session['is_compressed']:
            scratch_dir = os.path.join(scratch_dir, session_id)

            # check if this was previously extracted by us.
            if not os.path.exists(os.path.join(scratch_dir, ir_name)):
                session_path = os.path.join(the_session['directory'], the_session['session_id']+'.tar.gz')

                with ProgressFileObject(session_path) as pfo:
                    pfo.progress.disable = True
                    pfo.progress.set_description('Scanning tarball {}'.format(the_session['session_id']))
                    pfo.progress.leave = False
                    try:
                        tar = tarfile.open(fileobj=pfo, mode='r:gz')
                        tar_members = tar.getmembers()
                        tar_names = [_.name for _ in tar_members]
                        to_extract = [
                            tar_members[tar_names.index(ir_name)],
                            tar_members[tar_names.index(ir_ts_name)]
                        ]
                        ensure_dir(scratch_dir)
                        pfo.progress.set_description('Extracting')
                        pfo.progress.reset(total=sum([m.size for m in to_extract]))
                        tar.extractall(path=scratch_dir, members=to_extract)
                    except tarfile.TarError as e:
                        raise type(e)("{}: {}".format(the_session['session_id'], str(e))).with_traceback(sys.exc_info()[2])
            else:
                pass #sys.stderr.write('Found previously extracted; using that...\n')

            ir_path = os.path.join(scratch_dir, ir_name)
            ir_timestamp_path = os.path.join(scratch_dir, ir_ts_name)
        else:
            ir_path = os.path.join(the_session['directory'], ir_name)
            ir_timestamp_path = os.path.join(the_session['directory'], ir_ts_name)

        clip = VideoFileClip(ir_path, fps_source="tbr")

        # crop time to the specified slice (synching with depth data)
        ir_times = load_timestamps(ir_timestamp_path)
        depth_times = load_h5_timestamps(h5)[slice[0][0]:slice[0][1]]
        ir_time_start = (nearest(ir_times, depth_times[0]) - ir_times[0]) / 1000
        ir_time_end = (nearest(ir_times, depth_times[-1]) - ir_times[0]) / 1000
        clip = clip.subclip(ir_time_start, ir_time_end)

        if crop == 'none':
            pass
        elif crop == 'auto':
            crop_region = get_mask_bounds(h5)
            clip = moviepy_crop(clip, **crop_region)
        elif isinstance(crop, dict):
            clip = moviepy_crop(clip, **crop)

        return clip
#end fetch_ir_clip()


def get_mask_bounds(h5):
    mask = h5['/metadata/extraction/roi'][()]
    mask_idx = np.nonzero(mask)
    return {
        'x1': ensure_even(np.min(mask_idx[1])),
        'y1': ensure_even(np.min(mask_idx[0])),
        'x2': ensure_even(np.max(mask_idx[1])),
        'y2': ensure_even(np.max(mask_idx[0]))
    }
#end get_mask_bounds()

def ensure_even(num):
    if (num % 2) == 1:
        return num + 1
    else:
        return num
#end ensure_even()

def nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array-value)).argmin()
    return array[idx]
# nearest()




