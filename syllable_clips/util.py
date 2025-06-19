

import argparse
import io
import json
import os
from typing import IO, Dict, Union
from typing_extensions import Literal, TypedDict
from moseq2_viz.model.util import parse_model_results, relabel_by_usage


import numpy as np


class LabelMapping(TypedDict):
    raw: int
    usage: int
    frames: int

LabelMap = Dict[int, LabelMapping]
def get_syllable_id_mapping(model_file: str) -> LabelMap:
    ''' Gets a mapping of syllable IDs

    Parameters:
        model_file (str): path to a model to interrogate

    Returns:
        dict of dicts, indexed by raw id, where each sub-dict contains raw, usage, and frame ID assignments
    '''
    mdl = parse_model_results(model_file, sort_labels_by_usage=False)
    labels_usage = relabel_by_usage(mdl['labels'], count='usage')[1]
    labels_frames = relabel_by_usage(mdl['labels'], count='frames')[1]

    available_ids = list(set(labels_usage + labels_frames))
    label_map: LabelMap = {i: {'raw': i, 'usage': -1, 'frames': -1} for i in available_ids}
    label_map[-5] = {'raw': -5, 'usage': -5, 'frames': -5}  # -5 is the "unknown" label

    for usage_id, raw_id in enumerate(labels_usage):
        label_map[raw_id]['usage'] = usage_id

    for frames_id, raw_id in enumerate(labels_frames):
        label_map[raw_id]['frames'] = frames_id

    return label_map


def reindex_label_map(label_map: LabelMap, by: Literal['usage', 'frames', 'raw']) -> LabelMap:
    ''' Reindex a label map by usage, frames, or raw ID

    Parameters:
        label_map (LabelMap): The label map to reindex
        by (str): The key to reindex by, one of {'usage', 'frames', 'raw'}

    Returns:
        LabelMap: A new label map indexed by the specified key
    '''
    if by not in ['usage', 'frames', 'raw']:
        raise ValueError(f"Invalid index type '{by}'. Must be one of ['usage', 'frames', 'raw']")

    return {itm[by]: itm for itm in label_map.values()}


class NumpyEncoder(json.JSONEncoder):
    ''' Special json encoder for numpy types '''
    def default(self, obj): # pylint: disable=method-hidden
        np_int_types = (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32,
                    np.int64, np.uint8, np.uint16, np.uint32, np.uint64)
        np_flt_types = (np.float_, np.float16, np.float32, np.float64)
        if isinstance(obj, np_int_types):
            return int(obj)
        elif isinstance(obj, np_flt_types):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)): #### This is the fix
            return obj.tolist()
        else:
            return json.JSONEncoder.default(self, obj)
#end class NumpyEncoder

def get_max_states(model_file: str) -> int:
    ''' Gets the maximum number of states parameter from model training.
        This corresponds to the `--max-states` parameter from `moseq2-model learn-model` command.

        Parameters:
            model_file (str): path to the model file to interrogate
        
        Returns:
            int: max number of states parameter from model training
    '''
    model = parse_model_results(model_file)
    return model['run_parameters']['max_states']

def dir_path_arg(path: str) -> str:
    ''' Argparse type parser, ensuring the argument is a directory that exists
    '''
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path to a readable directory")


def load_timestamps(timestamp_file: Union[str, IO[bytes]], col: int = 0) -> np.ndarray:
    """Read timestamps from space delimited text file.

    Args:
        timestamp_file (str): path to a file containing timestamp data
        col (int): column of the file containing timestamp data

    Returns:
    np.ndarray containing timestamp data.
    """
    timestamps = []
    if isinstance(timestamp_file, str):
        with open(timestamp_file, "r", encoding="utf-8") as ts_file:
            for line_str in ts_file:
                cols = line_str.split()
                timestamps.append(float(cols[col]))
        return np.array(timestamps)
    elif isinstance(timestamp_file, io.BufferedReader):
        # try iterating directly
        for line_bytes in timestamp_file:
            cols = line_bytes.decode().split()
            timestamps.append(float(cols[col]))
        return np.array(timestamps)
    else:
        raise ValueError("Could not understand parameter timestamp_file!")