import argparse
import os
from re import sub
import shutil
import sys

import numpy as np
import psutil

from syllable_clips.session import ensure_unpacked_sessions
from syllable_clips.syllable_clips import produce_clips
from syllable_clips.util import dir_path_arg, get_max_states, get_syllable_id_mapping, reindex_label_map
from syllable_clips import __version__


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    subparsers = parser.add_subparsers()

    single_parser = subparsers.add_parser('single', help="Render just one example of one given syllable")
    single_parser.add_argument('syllable_id', type=int)
    single_parser.set_defaults(func=single_example)

    sngmul_parser = subparsers.add_parser('single-multiple', help="Render multiple examples of one given syllable")
    sngmul_parser.add_argument('syllable_id', type=int)
    sngmul_parser.add_argument('--num-examples', type=int, default=10, help="Number of examples for each syllable to render")
    sngmul_parser.set_defaults(func=single_multiple_examples)

    corpus_parser = subparsers.add_parser('corpus', help="Render just one example of each syllable")
    corpus_parser.set_defaults(func=corpus_single_example)

    corpmulti_parser = subparsers.add_parser('corpus-multiple', help="Render multiple examples of each syllable")
    corpmulti_parser.add_argument('--num-examples', type=int, default=10, help="Number of examples for each syllable to render")
    corpmulti_parser.set_defaults(func=corpus_multiple_examples)

    for _, subp in subparsers.choices.items():
        subp.formatter_class = argparse.ArgumentDefaultsHelpFormatter
        subp.add_argument('index')
        subp.add_argument('model')
        subp.add_argument('--sort', action='store_true', help="Sort syllables")
        subp.add_argument('--count', choices=['usage', 'frames'], default='usage', help="Metric for sorting")

        subp.add_argument('--name', default="syllable_clip", help="Output filename prefix")
        subp.add_argument('--dir', help="Output directory")
        subp.add_argument('-r', '--raw-path', type=dir_path_arg, help="Location of the raw data. Needed if using rgb stream.")
        subp.add_argument('--manifest', help="For older extractions, provide a manifest file mapping h5 uuid to session id")
        subp.add_argument('--man-uuid-col', default='UUID')
        subp.add_argument('--man-session-id-col', default='Session_ID')
        subp.add_argument('--fps', default=30, help="Frames per second")
        subp.add_argument('--scratch', help="Scratch location used for possible extraction")
        #subp.add_argument('--no-cleanup', action='store_false', dest='cleanup', help="Scratch location used for possible extraction")
        subp.add_argument('--cleanup', action='store_true', dest='cleanup', help="Cleanup scratch directory after processing")
        subp.add_argument('--crop-rgb', action='store', default='auto', help="Crop the rgb to the bounds of the extracted region. Auto crops to the ROI from extraction (old formats not supported). None will not crop. Or supply a list of coordinates as 'x1,y1,x2,y2' ex: '20,50,100,150'")
        subp.add_argument('--crop-ir', action='store', default='auto', help="Crop the IR to the bounds of the extracted region. Auto crops to the ROI from extraction (old formats not supported). None will not crop. Or supply a list of coordinates as 'x1,y1,x2,y2' ex: '20,50,100,150'")

        subp.add_argument('--rgb-name', action='store', default='rgb.mp4', help="Name of the rgb video file within the raw data directory")
        subp.add_argument('--rgb-ts-name', action='store', default='rgb_ts.txt', help="Name of the rgb video timestamp file within the raw data directory")
        subp.add_argument('--ir-name', action='store', default='ir.mp4', help="Name of the ir video file within the raw data directory")
        subp.add_argument('--ir-ts-name', action='store', default='ir_ts.txt', help="Name of the ir video timestamp file within the raw data directory")

        pick_choices = ['median', 'longest', 'shortest', 'shuffle']
        subp.add_argument('--pick', choices=pick_choices, default=pick_choices[0], help="Method for choosing which syllable instance(s) to plot.")
        stream_choices = ['depth', 'rgb', 'ir', 'composed']
        subp.add_argument('--streams', choices=stream_choices, nargs="+", help="Video stream(s) to use.")

        subp.add_argument('--prepend', default=0.0, type=float, help="Amount of time to prepend to the syllable slice in seconds")
        subp.add_argument('--append', default=0.0, type=float, help="Amount of time to append to the syllable slice in seconds")

        subp.add_argument('--gaussfilter-space', default=(0, 0), nargs=2, type=float, help="Spatial filter for data (Gaussian)")
        subp.add_argument('--medfilter-space', default=[0], type=int, action="append", help="Median spatial filter")
        subp.add_argument('--min-height', type=int, default=5, help="Minimum height for scaling videos")
        subp.add_argument('--max-height', type=int, default=80, help="Minimum height for scaling videos")
        subp.add_argument('--cmap', type=str, default='jet', help="Name of valid Matplotlib colormap for false-coloring images")

        # NOTE: cpu_affinity does not work on macOS systems, so we use cpu_count instead. This shouldn't be an issue because slurm
        # does not get utilized on macOS, so cpu_count is sufficient as there will never be a case in which we are asking for more
        # cpu cores than are allocated to us.
        #           - Jared @ 24 May 2021
        num_processors = psutil.cpu_count() if sys.platform == 'darwin' else len(psutil.Process().cpu_affinity())
        subp.add_argument('-p', '--processors', default=num_processors, type=int, help="Number of CPUs to use")

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    sys.stderr.write("Using {} processors to do work.\n".format(args.processors))

    if args.dir is None:
        args.dir = os.path.join(os.path.dirname(os.path.abspath(args.index)), args.name)
    os.makedirs(args.dir, exist_ok=True)

    if args.scratch is None:
        args.scratch = os.path.join(args.raw_path, 'scratch')


    to_extract = ['metadata.json']
    if "rgb" in args.streams:
        to_extract.extend([args.rgb_name, args.rgb_ts_name])
    if "ir" in args.streams:
        to_extract.extend([args.ir_name, args.ir_ts_name])
    ensure_unpacked_sessions(args.raw_path, args.scratch, to_extract)

    # Scratch directory might not be created, if no sessions were unpacked.
    # If it exists, we can add it to the raw_path list.
    if os.path.exists(args.scratch):
        args.raw_path = [args.raw_path, args.scratch]

    label_map = get_syllable_id_mapping(args.model)
    if args.sort:
        args.label_map = reindex_label_map(label_map, by=args.count)


    args.crop_rgb = parse_crop_arg(args.crop_rgb)
    args.crop_ir = parse_crop_arg(args.crop_ir)


    # run the command
    args.func(args)

    if args.cleanup:
        shutil.rmtree(args.scratch)
#end main()

def parse_crop_arg(arg):
    ''' Parses the crop argument for rgb and ir streams.
    
    Parameters:
        arg (str): Argument string to parse, should be 'auto', 'none', or 'x1,y1,x2,y2'
    
    Returns:
        dict: Dictionary with keys 'x1', 'y1', 'x2', 'y2' if coordinates are provided, otherwise returns 'auto' or 'none'.
    '''
    if arg.lower() == 'none':
        return 'none'
    elif arg.lower() == 'auto':
        return 'auto'
    else:
        coords = [int(x) for x in arg.split(',')]
        if len(coords) != 4:
            raise RuntimeError('Argument for crop in valid! must be one of "auto", "none", or list of coordinates "x1,y1,x2,y2"')
        return { k: coords[i] for i, k in enumerate(['x1','y1','x2','y2']) }


def single_example(args):
    produce_clips([args.syllable_id], 1, args)
#end single_example()

def single_multiple_examples(args):
    produce_clips([args.syllable_id], args.num_examples, args)
#end single_multiple_examples()

def corpus_single_example(args):
    max_states = get_max_states(args.model)
    produce_clips(np.arange(0, max_states), 1, args)
#end corpus_single_example()

def corpus_multiple_examples(args):
    max_states = get_max_states(args.model)
    produce_clips(np.arange(0, max_states), args.num_examples, args)
#end corpus_multiple_examples()

if __name__ == '__main__':
    main()