import random
import re
from datetime import timedelta
from typing import Callable, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
from moseq2_viz.model.util import get_syllable_slices, parse_model_results
from moseq2_viz.util import parse_index
from typing_extensions import Literal, TypedDict


def load_h5_timestamps(h5: h5py.File, ts_path: Optional[str]=None) -> np.ndarray:
    ''' Load timestamps from an h5 file, looking in known timestamp locations

        Parameters:
            h5 (h5py.File): h5 file to load timestamps from
            ts_path (str): Path to the timestamp dataset. If None, try loading from known paths.

        Returns:
            ndarray: timestamp data
    '''
    if ts_path is None:
        to_search = ['/timestamps', '/metadata/timestamps']
        for s in to_search:
            if s in h5:
                ts_path = s
                break

    if ts_path not in h5:
        raise RuntimeError('Could not find timestamps for file {}'.format(h5.filename))

    return h5[ts_path][()]
#end load_timestamps()


Slice = Tuple[Tuple[int, int], str, str]
""" A slice is a tuple of the form ((start, end), uuid, h5_path)."""

class SliceInfo(TypedDict):
    session_id: str      # session ID
    uuid: str            # extraction UUID
    onset_idx: int       # index of the start of emission
    offset_idx: int      # index of the stop of emission
    onset_time: str      # Wall clock time of the start of emission relative to beginning of recording
    offset_time: str     # Wall clock time of the stop of emission relative to beginning of recording
    start_idx: int       # index of the start of slice (possibly expanded)
    end_idx: int         # index of the end of slice (possibly expanded)
    start_time: str      # Wall clock time of the start of slice (possibly expanded) relative to beginning of recording
    end_time: str        # Wall clock time of the end of slice (possibly expanded) relative to beginning of recording
    duration: float      # duration of the slice in seconds

def expand_slice(slice: Slice, t_prepend: float=0.0, t_append: float=0.0, manifest_path: Optional[str]=None,
                 manifest_session_id_col: Optional[str]=None, manifest_uuid_col: Optional[str]=None) -> Tuple[Slice, SliceInfo]:
    ''' Expands a slice by the specified amount of time. Returned slice 
        is safe with regard to the [index] bounds of the data

        If your h5 files do not store the session id information (usually located at the path
        `'/metadata/extraction/parameters/input_file'`), it can be retrieved using a manifest file containing
        the session id as well as uuid for the given h5. In this case utilize the `manifest_path`,
        `manifest_session_id_col` and `manifest_uuid_col` parameters to define this mapping.

        Parameters:
            slice: (slice) - The slice to operate upon
            t_prepend: float - amount of time in seconds to prepend to this slice
            t_append: float - amount of time in seconds to prepend to this slice
            manifest_path (str): path to manifest file
            manifest_session_id_col (str): name of the column containing session ids within manifest
            manifest_uuid_col (str): name of the column containing UUIDs within manifest

        Returns:
            (slice, info_dict) - the expanded slice, and dict with info about the slice
    '''
    with h5py.File(slice[2], 'r') as h5:
        all_timestamps = load_h5_timestamps(h5)
        ts_length = len(all_timestamps)

        p_offset = 0
        p_start = e_start = all_timestamps[slice[0][0]]
        if t_prepend > 0:
            while e_start - p_start < t_prepend * 1000 and slice[0][0] + p_offset > 0:
                p_offset -= 1
                p_start = all_timestamps[slice[0][0] + p_offset]


        a_offset = 0
        a_end = e_end = all_timestamps[slice[0][1]]
        if t_append > 0:
            while (a_end - e_end < t_prepend * 1000) and (slice[0][1] + a_offset < ts_length-1):
                a_offset += 1
                a_end = all_timestamps[slice[0][1] + a_offset]

        info: SliceInfo = {
            'session_id': get_session_id(h5, manifest_path=manifest_path, manifest_session_id_col=manifest_session_id_col, manifest_uuid_col=manifest_uuid_col),
            'uuid': h5['/metadata/uuid'][()],
            'onset_idx': slice[0][0],
            'offset_idx': slice[0][1],
            'onset_time': str(timedelta(seconds=(e_start - all_timestamps[0]) / 1000)),
            'offset_time': str(timedelta(seconds=(e_end - all_timestamps[0]) / 1000)),
            'start_idx': slice[0][0] + p_offset,
            'end_idx': slice[0][1] + a_offset,
            'start_time': str(timedelta(seconds=(p_start - all_timestamps[0]) / 1000)),
            'end_time': str(timedelta(seconds=(a_end - all_timestamps[0]) / 1000)),
            'duration': (all_timestamps[slice[0][1]] - all_timestamps[slice[0][0]]) / 1000
        }

        slice = ((slice[0][0] + p_offset, slice[0][1] + a_offset), slice[1], slice[2])
    return (slice, info)
#end prepare_slice()

CountMode = Literal['usage', 'frames']
def prep_slice_data(model_file: str, index_file: str, sort_labels: bool=True, count: CountMode='usage') -> Callable[[int, "SortMethod"], List[Slice]]:
    ''' From a model and index, prepare slices

    Parameters:
        model_file: (string) model file to use
        index_file: (string) index file to use
        sort_labels: (bool) sort labels by `count` or not
        count: (string) method used for counting/sorting

    Returns:
        function (sid, method) => slices[]
    '''
    index, sorted_index = parse_index(index_file)
    model = parse_model_results(model_file, sort_labels_by_usage=sort_labels, count=count)

    def fetch_slices(sid: int, sort_method: SortMethod):
        slices = get_syllable_slices(sid, model['labels'], model['keys'], sorted_index)
        return sort_slices(slices, sort_method)

    return fetch_slices
#end prep_data()

SortMethod = Literal['median', 'longest', 'shortest', 'shuffle']
def sort_slices(slices: List[Slice], method: SortMethod) -> List[Slice]:
    ''' Sorts `slices` according to `method`

    Parameters:
        slices: (list) List of slices
        method: (string) Method used to sort slices. One of {median-length, longest, shortest, shuffle}

    Returns:
        list: slices sorted according to `method`
    '''
    if method == 'median':
        #sort by distance to the median duration
        median_dur = np.nanmedian([s[0][1]-s[0][0] for s in slices])
        return sorted(slices, key=lambda x: abs(median_dur - (x[0][1]-x[0][0])), reverse=False)

    elif method == 'longest':
        #sort by Longest to Shortest duration
        return sorted(slices, key=lambda x: (x[0][1]-x[0][0]), reverse=True)

    elif method == 'shortest':
        #sort by Shortest to Longest
        return sorted(slices, key=lambda x: (x[0][1]-x[0][0]), reverse=False)

    elif method == 'shuffle':
        random.shuffle(slices)
        return slices

    else:
        raise ValueError("Method {} not supported!".format(method))
#end pick_slice()


def get_session_id(h5: h5py.File, pattern: str=r'session_\d+', manifest_path: Optional[str]=None,
                   manifest_session_id_col: Optional[str]=None, manifest_uuid_col: Optional[str]=None) -> str:
    ''' Retrieves the session id that `h5` was derived from

        Internally searches the H5 key '/metadata/extraction/parameters/input_file'

        If your h5 files do not store this information, it can be retrieved using a manifest file containing
        the session id as well as uuid for the given h5. In this case utilize the `manifest_path`,
        `manifest_session_id_col` and `manifest_uuid_col` parameters to define this mapping.

        Parameters:
            h5: (h5py.File): H5 file to inspect
            pattern: (regex pattern): pattern used to extract session id
            manifest_path (str): path to manifest file
            manifest_session_id_col (str): name of the column containing session ids within manifest
            manifest_uuid_col (str): name of the column containing UUIDs within manifest

        Returns:
            string: session ID
    '''
    input_file_path = '/metadata/extraction/parameters/input_file'
    if input_file_path in h5 and manifest_path is None:
        in_file = h5[input_file_path][()]
        if not isinstance(in_file, str):
            in_file = in_file.decode('utf-8') # might be bytes!
        session_pattern = re.compile(pattern)
        match = session_pattern.search(in_file)
        if match is None:
            raise RuntimeError(f"Could not find session ID in {in_file} using pattern {pattern}")
        session_id = match.group()
    else:
        manifest = pd.read_csv(manifest_path, sep='\t')
        uuid = h5['/metadata/uuid'][()]
        session_id = manifest[manifest[manifest_uuid_col] == uuid][manifest_session_id_col].iloc[0]
    return session_id
#end getSessionId()
