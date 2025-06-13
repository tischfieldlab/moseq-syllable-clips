
from datetime import timedelta
from itertools import groupby
import logging
from operator import itemgetter
import os
import subprocess
from typing import Iterable, List, Optional, Sequence, Tuple, TypeVar, Union
from matplotlib import pyplot as plt
import numpy as np
import tqdm
from typing_extensions import TypedDict, Literal


T = TypeVar('T', int, float)
def collapse_consecutive_values(values: Iterable[T]) -> List[Tuple[T, int]]:
    '''Collapses consecutive values in an array.

    Example:
    collapse_consecutive_values([0,1,2,3,10,11,12,13,21,22,23])
    > [(0,4), (10,4), (21, 3)]

    Args:
    a (np.ndarray): array of labels to collapse

    Returns
    List[Tuple[float, int]]: each tuple contains (seed, run_count)
    '''
    grouped_instances = []
    for _, group in groupby(enumerate(values), lambda ix : ix[0] - ix[1]):
        local_group = list(map(itemgetter(1), group))
        grouped_instances.append((local_group[0], len(local_group)))
    return grouped_instances
#end collapse_adjacent_values()


class FFProbeInfo(TypedDict):
    ''' Represents the results of an ffprobe call.'''
    file: str
    codec: str
    pixel_format: str
    dims: Tuple[int, int]
    fps: float
    nframes: int

class VideoReader:
    def __init__(self, video_path: str, out_pixel_format: Literal['gray16le', 'rgb24'] = 'rgb24', threads: int = 2, slices: int = 24, slicecrc: int = 1):
        """ Initialize the video reader.
        Args:
            video_path (str): path to the video file
            out_pixel_format (Literal['gray16le', 'rgb24']): pixel format to use for the returned pixels frames, defaults to 'rgb24'
            threads (int): number of threads to use for reading frames
            slices (int): number of slices to use for reading frames
            slicecrc (int): check integrity of slices, defaults to 1
        """
        self.video_path = video_path
        self.out_pixel_format = out_pixel_format
        self.threads = threads
        self.slices = slices
        self.slicecrc = slicecrc
        self.info: FFProbeInfo = self._get_video_info()

    def _get_video_info(self) -> FFProbeInfo:
        '''Get information about this video via ffmpeg's ffprobe utility.

        Returns:
            FFProbeInfo - dict containing information about video `filename`
        '''
        command = [
            'ffprobe',
            '-v', 'fatal',
            '-show_entries',
            'stream=width,height,r_frame_rate,nb_frames,codec_name,pix_fmt',
            '-of',
            'default=noprint_wrappers=1:nokey=1',
            self.video_path,
            '-sexagesimal'
        ]

        ffmpeg = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        out, err = ffmpeg.communicate()

        if err:
            logging.error(err, stack_info=True)
        out_lines = out.decode().split(os.linesep)

        return {
            'file': self.video_path,
            'codec': out_lines[0],
            'pixel_format': out_lines[3],
            'dims': (int(out_lines[1]), int(out_lines[2])),
            'fps': float(out_lines[4].split('/')[0])/float(out_lines[4].split('/')[1]),
            'nframes': int(out_lines[5])
        }

    def read_frames(self, start_time: Optional[float] = None, stop_time: Optional[float] = None) -> np.ndarray:
        '''Reads in frames from a video file using a pipe from ffmpeg.

        Args:
            start_time (Optional[float]): start time in seconds to read frames from, defaults to 0.0
            stop_time (Optional[float]): stop time in seconds to read frames to, defaults to the end of the video

        Returns:
            3d numpy array:  frames x h x w
        '''
        if start_time is None:
            start_time = 0.0
        if stop_time is None:
            stop_time = self.info['nframes'] / self.info['fps']

        nframes = int((stop_time - start_time) * self.info['fps'])

        #print(f'Reading {nframes} frames from {self.video_path} starting at {timedelta(seconds=start_time)} and ending at {timedelta(seconds=stop_time)}')

        frame_size = self.info['dims']

        out_shape: Union[Tuple[int, int, int], Tuple[int, int, int, int]]
        if self.out_pixel_format == 'gray16le':
            dtype = 'uint16'
            out_shape = (nframes, frame_size[1], frame_size[0])
        elif self.out_pixel_format == 'rgb24':
            dtype = 'uint8'
            out_shape = (nframes, frame_size[1], frame_size[0], 3)

        command = [
            'ffmpeg',
            '-loglevel', 'fatal',
            '-ss', str(start_time),
            '-i', self.video_path,
            '-vframes', str(nframes),
            '-f', 'image2pipe',
            '-s', f'{frame_size[0]:d}x{frame_size[1]:d}',
            '-pix_fmt', self.out_pixel_format,
            '-threads', str(self.threads),
            '-slices', str(self.slices),
            '-slicecrc', str(self.slicecrc),
            '-vcodec', 'rawvideo',
            '-'
        ]

        pipe = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        out, err = pipe.communicate()

        if err:
            raise RuntimeError(err.decode())

        return np.copy(np.frombuffer(out, dtype=dtype).reshape((nframes, *out_shape[1:])))



class VideoWriter():
    ''' Encapsulate state needed for generating Preview Videos
    '''
    def __init__(self, filename: str, fps: int = 30, tqdm_opts: Optional[dict] = None, threads: int = 2,
                 pixel_format: str = 'rgb24', codec: str = 'h264', slices: int = 24, slicecrc: int = 1) -> None:
        """ Initialize the video writer.
        
        Args:
            filename (str): path to the output video file
            fps (int): frames per second for the output video
            tqdm_opts (Optional[dict]): options for tqdm progress bar, defaults to None
            threads (int): number of threads to use for encoding
            pixel_format (str): pixel format for the output video, defaults to 'rgb24'
            codec (str): codec to use for encoding, defaults to 'h264'
            slices (int): number of slices to use for encoding, defaults to 24
            slicecrc (int): check integrity of slices, defaults to 1
        """
        self.filename = filename
        self.fps = fps
        self.threads = threads
        self.pixel_format = pixel_format
        self.codec = codec
        self.slices = slices
        self.slicecrc = slicecrc
        self.video_pipe: Union[subprocess.Popen[bytes], None] = None
        self.tqdm_opts = {
            'leave': False,
            'disable': True,
        }
        if tqdm_opts is not None:
            self.tqdm_opts.update(tqdm_opts)

    def write_frames(self, frames: np.ndarray):
        ''' Write frames to the preview video

        Args:
            frames (np.ndarray): frames to render
        '''
        if not np.mod(frames.shape[1], 2) == 0:
            frames = np.pad(frames, ((0, 0), (0, 1), (0, 0), (0, 0)), 'constant', constant_values=0)

        if not np.mod(frames.shape[2], 2) == 0:
            frames = np.pad(frames, ((0, 0), (0, 0), (0, 1), (0, 0)), 'constant', constant_values=0)

        command = [
            'ffmpeg',
            '-y',
            '-loglevel', 'fatal',
            '-threads', str(self.threads),
            '-framerate', str(self.fps),
            '-f', 'rawvideo',
            '-s', f'{frames.shape[2]:d}x{frames.shape[1]:d}',
            '-pix_fmt', self.pixel_format,
            '-i', '-',
            '-an',
            '-vcodec', self.codec,
            '-slices', str(self.slices),
            '-slicecrc', str(self.slicecrc),
            '-r', str(self.fps),
            '-pix_fmt', 'yuv420p',
            '-tune', 'zerolatency',
            '-preset', 'ultrafast',
            self.filename
        ]

        if self.video_pipe is None:
            self.video_pipe = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        assert self.video_pipe.stdin is not None, "Video pipe stdin is None, cannot write frames!"

        for i in tqdm.tqdm(range(frames.shape[0]), desc="Writing frames", **self.tqdm_opts):
            self.video_pipe.stdin.write(frames[i, ...].tobytes())

    def close(self):
        ''' Close the video writer
        '''
        if self.video_pipe:
            self.video_pipe.communicate()


def stack_videos(videos: Sequence[np.ndarray], orientation: Literal['horizontal', 'vertical', 'diagional']) -> np.ndarray:
    ''' Stack videos according to orientation to create one big video

    Args:
        videos (Iterable[np.ndarray]): Iterable of videos to stack of shape (nframes, height, width, channels).
            All videos must match dimentions in axis 0 and axis 3
        orientation (Literal['horizontal', 'vertical', 'diagional']): orientation of stacking.

    Retruns:
        stacked composite video
    '''
    dtype = reduce_dtypes(videos)
    nframes = reduce_axis_size(videos, 0)
    channels = reduce_axis_size(videos, 3)
    heights = [v.shape[1] for v in videos]
    widths = [v.shape[2] for v in videos]

    if orientation == 'horizontal':
        height = max(heights)
        width = sum(widths)
    elif orientation == 'vertical':
        height = sum(heights)
        width = max(widths)
    elif orientation == 'diagional':
        height = sum(heights)
        width = sum(widths)
    else:
        raise ValueError(f'Unknown orientation "{orientation}". Expected one of ["horizontal", "vertical"].')

    output_movie = np.zeros((nframes, height, width, channels), dtype)
    for i, video in enumerate(videos):
        if orientation == 'horizontal':
            offset = sum([w for wi, w in enumerate(widths) if wi < i])
            output_movie[:, :heights[i], offset:offset+widths[i], :] = video
        elif orientation == 'vertical':
            offset = sum([h for hi, h in enumerate(heights) if hi < i])
            output_movie[:, offset:offset+heights[i], :widths[i], :] = video
        elif orientation == 'diagional':
            offsetw = sum([w for wi, w in enumerate(widths) if wi < i])
            offseth = sum([h for hi, h in enumerate(heights) if hi < i])
            output_movie[:, offseth:offseth+heights[i], offsetw:offsetw+widths[i], :] = video

    return output_movie


def reduce_axis_size(data: Sequence[np.ndarray], axis: int) -> int:
    ''' Reduce an iterable of numpy.ndarrays to a single scalar for a given axis
    Will raise an exception if no items are passed or the arrays are not the same size on the given axis

    Args:
        data (Iterable[np.ndarray]): Arrays to inspect
        axis (int): axis to be inspected

    Returns:
        int - the shared shape of `axis` in all arrays in `data`
    '''
    if len(data) <= 0:
        raise ValueError('Need a list with at least one array!')

    sizes = {d.shape[axis] for d in data}
    if len(sizes) == 1:
        return int(sizes.pop())
    else:
        raise ValueError(f'Arrays should be equal sized on axis{axis}! Got arrays with shapes {[d.shape for d in data]}.')


def reduce_dtypes(data: Sequence[np.ndarray]) -> np.dtype:
    ''' Reduce an iterable of numpy.ndarrays to a dtype
    Will raise an exception if no items are passed or the all arrays do not share a dtype

    Args:
        data (Iterable[np.ndarray]): Arrays to inspect

    Returns:
        npt.DTypeLike - the shared dtype of all arrays in `data`
    '''
    if len(data) <= 0:
        raise ValueError('Need a list with at least one array!')

    dtypes = {d.dtype for d in data}
    if len(dtypes) == 1:
        return dtypes.pop()
    else:
        raise ValueError(f'Arrays should have same dtype! Got dtypes: {[d.dtype for d in data]}')


def colorize_video(frames: np.ndarray, vmin: float=0, vmax: float=100, cmap: str='jet') -> np.ndarray:
    ''' Colorize single channel video data

    Args:
        frames (np.ndarray): frames to be colorized, assumed shape (nframes, height, width)
        vmin (float): minimum data value corresponding to cmap min
        vmax (float): maximum data value corresponding to cmap max
        cmap (str): colormap to use for converting to color

    Returns:
        np.ndarray containing colorized frames of shape (nframe, height, width, 3)
    '''
    use_cmap = plt.get_cmap(cmap)

    disp_img = frames.copy().astype('float32')
    disp_img = (disp_img-vmin)/(vmax-vmin)
    disp_img[disp_img < 0] = 0
    disp_img[disp_img > 1] = 1
    disp_img = use_cmap(disp_img)[...,:3]*255

    return disp_img.astype('uint8')


class CropRegion(TypedDict):
    ''' Represents a crop argument for cropping video frames '''
    x1: int
    y1: int
    x2: int
    y2: int
def crop_video(frames: np.ndarray, crop: CropRegion) -> np.ndarray:
    ''' Crop video frames according to crop argument

    Args:
        frames (np.ndarray): frames to be cropped, assumed shape (nframes, height, width, channels)
        crop (CropRegion): crop argument, can be 'auto', 'none', or a dict with keys x1, y1, x2, y2

    Returns:
        np.ndarray containing cropped frames of shape (nframe, height, width, channels)
    '''
    return frames[:, crop['y1']:crop['y2'], crop['x1']:crop['x2'], :]


def draw_onset_indicator(frames: np.ndarray, onset: int, offset: int, size: Tuple[int, int], color: Optional[Tuple[int, int, int]]=None) -> np.ndarray:
    ''' Draw an onset indicator on the first frame of the video

    Args:
        frames (np.ndarray): frames to draw on, assumed shape (nframes, height, width, channels)
        onset (int): frame index to start drawing the indicator
        offset (int): frame index to stop drawing the indicator
        size (Tuple[int, int]): size of the indicator in pixels (height, width)
        color (Tuple[int, int, int]): color of the indicator as an RGB tuple, defaults to red (255, 0, 0) if None

    Returns:
        np.ndarray containing frames with onset indicator drawn
    '''
    if onset < 0 or onset >= frames.shape[0]:
        raise ValueError(f'Onset {onset} is out of bounds for video with {frames.shape[0]} frames!')
    if offset < 0 or offset > frames.shape[0]:
        raise ValueError(f'Offset {offset} is out of bounds for video with {frames.shape[0]} frames!')
    if onset >= offset:
        raise ValueError(f'Onset {onset} must be less than offset {offset}!')

    if color is None:
        color = (255, 0, 0)

    half_x = size[0] / 2
    half_y = size[1] / 2
    r = min(half_x, half_y)

    xx, yy = np.mgrid[:size[0], :size[1]]
    circle_dist = (xx - half_x) ** 2 + (yy - half_y) ** 2
    to_draw = (circle_dist < r**2).nonzero()

    frames[onset:offset+1, to_draw[0], to_draw[1], :] = color

    return frames
