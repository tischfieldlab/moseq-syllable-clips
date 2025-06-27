


import io
import logging
import os
import re
import sys
import tarfile
from typing import List, Optional, Union
from typing_extensions import TypedDict

import tqdm


class SessionInfo(TypedDict):
    is_compressed: bool
    session_id: str
    directory: str
    metadata: str

def find_sessions_path(root_dir: Optional[Union[str, List[str]]]=None, recursive: bool=True,
                       skip_uncompressed: bool=False, uncompressed_pattern: str=r'metadata\.json',
                       skip_compressed: bool=False, compressed_pattern: str=r'session_\d+\.json') -> List[SessionInfo]:
    ''' Finds sessions (raw data) on the path root_dir

    Parameters:
    root_dir (string): Path to search for sessions, if None then search the current working directory.
    recursive (bool): Recursively search root_dir for sessions
    skip_uncompressed (bool): Skip (do not return) uncompressed sessions if True
    uncompressed_pattern (string): Regular expression to match uncompressed session metadata files
    skip_compressed (bool): Skip (do not return) uncompressed sessions if True
    compressed_pattern (string): Regular expression to match compressed session metadata files

    Returns:
        sessions (List[SessionInfo]): List of session info dicts
    '''
    dirs_to_search = []
    if root_dir is None:
        dirs_to_search.append(os.getcwd())
    elif isinstance(root_dir, str):
        dirs_to_search.append(root_dir)
    elif isinstance(root_dir, list):
        dirs_to_search.extend(root_dir)


    unc_pattern = re.compile(uncompressed_pattern)
    cmp_pattern = re.compile(compressed_pattern)
    session_pattern = re.compile(r'session_\d+')
    results: List[SessionInfo] = []
    for dir_path in dirs_to_search:
        if not os.path.exists(dir_path):
            logging.warning("Path {} does not exist, skipping.".format(dir_path))
            continue
        if os.path.isfile(dir_path):
            # We got a file path directly... try to handle this
            # if it is a file, it is likely a compressed session
            if dir_path.endswith('.tar.gz') and not skip_compressed:
                m = session_pattern.search(dir_path)
                if m is None:
                    raise RuntimeError("Could not find session ID in {} using pattern {}".format(dir_path, session_pattern.pattern))
                results.append({
                    'is_compressed': True,
                    'session_id': m.group(0),
                    'directory': os.path.dirname(dir_path),
                    'metadata': os.path.join(os.path.dirname(dir_path), "{}.json".format(m.group(0)))
                })
        else:
            for item in os.listdir(dir_path):
                curr = os.path.join(dir_path, item)

                if os.path.isdir(curr) and (session_pattern.fullmatch(item) or recursive):
                    results.extend(find_sessions_path(curr,
                                                    recursive=recursive,
                                                    skip_compressed=skip_compressed,
                                                    skip_uncompressed=skip_uncompressed,
                                                    uncompressed_pattern=uncompressed_pattern, 
                                                    compressed_pattern=compressed_pattern))

                elif os.path.isfile(curr):
                    #check if we have 
                    if unc_pattern.fullmatch(item) is not None and not skip_uncompressed:
                        results.append({
                            'is_compressed': False,
                            'session_id': os.path.basename(dir_path),
                            'directory': dir_path,
                            'metadata': curr
                        })
                    elif cmp_pattern.fullmatch(item) is not None and not skip_compressed:
                        results.append({
                            'is_compressed': True,
                            'session_id': os.path.splitext(item)[0],
                            'directory': dir_path,
                            'metadata': curr
                        })
    return deduplicate_sessions(results)


def deduplicate_sessions(sessions: List[SessionInfo]) -> List[SessionInfo]:
    '''Deduplicates a list of session info dicts by session_id, preferring those that are uncompressed over compressed.

    Args:
        sessions (List[SessionInfo]): List of session info dicts

    Returns:
        List[SessionInfo]: List of deduplicated session info dicts
    '''
    seen = set()
    uncompressed = [s for s in sessions if not s['is_compressed']]
    compressed = [s for s in sessions if s['is_compressed']]
    deduped = []
    for session in uncompressed:
        if session['session_id'] not in seen:
            seen.add(session['session_id'])
            deduped.append(session)

    for session in compressed:
        if session['session_id'] not in seen:
            seen.add(session['session_id'])
            deduped.append(session)
    return deduped

def find_session_by_id(session_id: str, root_dir: Union[str, List[str]]) -> Optional[SessionInfo]:
    ''' Finds a session by its ID in the path root_dir

    Parameters:
        session_id (str): Session ID to search for
        root_dir (str): Path to search for sessions, if None then search the current working directory.
        recursive (bool): Recursively search root_dir for sessions

    Returns:
        SessionInfo: Session info dict if found, None otherwise
    '''
    sessions = find_sessions_path(root_dir=root_dir, skip_compressed=True)
    for session in sessions:
        if session['session_id'] == session_id:
            return session
    return None


def unpack_session(session_info: SessionInfo, to_extract: List[str], output_dir: str) -> str:
    '''Unpacks a compressed session file into the output directory.

    Parameters:
        session_path (str): Path to the compressed session file
        output_dir (str): Directory to unpack the session into, if None then use current working directory

    Returns:
        str: Path to the unpacked session directory
    '''
    if session_info['is_compressed'] is False:
        raise ValueError("Session is not compressed, cannot unpack: {}".format(session_info['session_id']))

    scratch_dir = os.path.join(output_dir, session_info['session_id'])
    os.makedirs(scratch_dir, exist_ok=True)
    session_path = os.path.join(session_info['directory'], session_info['session_id']+'.tar.gz')

    with ProgressFileObject(session_path) as pfo:
        pfo.progress.disable = True
        pfo.progress.set_description('Scanning tarball {}'.format(session_info['session_id']))
        pfo.progress.leave = False
        try:
            tar = tarfile.open(fileobj=pfo, mode='r:gz')
            tar_members = tar.getmembers()
            tar_names = [_.name for _ in tar_members]
            members_to_extract: List[tarfile.TarInfo] = [tar_members[tar_names.index(name)] for name in to_extract]
            pfo.progress.set_description('Extracting')
            pfo.progress.reset(total=sum([m.size for m in members_to_extract]))
            tar.extractall(path=scratch_dir, members=members_to_extract)
        except tarfile.TarError as e:
            raise type(e)("{}: {}".format(session_info['session_id'], str(e))).with_traceback(sys.exc_info()[2])

    return scratch_dir


class ProgressFileObject(io.FileIO):
    """Class used to provide provide file read progress updates."""

    def __init__(self, path: str, *args, progress: Optional[tqdm.tqdm] = None, tqdm_kwargs: Optional[dict] = None, **kwargs):
        """Construct an instance of ProgressFileObject.

        Will display a tqdm progress bar which updates as the file is read.

        Args:
        path (string): Path of the file to open
        progress (tqdm.tqdm): An (optional) instance of tqdm. If None, one is constructed for you
        tqdm_kwargs (dict): kwargs passed to `tqdm.tqdm.__init__()` if no tqdm instance is passed as `progress`.
        *args: additional arguments passed to io.FileIO
        **kwargs: additional kwargs passed to io.FileIO
        """
        if tqdm_kwargs is None:
            tqdm_kwargs = {}

        self._total_size = os.path.getsize(path)
        self.progress: tqdm.tqdm
        if progress is not None:
            assert isinstance(progress, (tqdm.tqdm,))
            self.progress = progress
            self.is_progress_external = True
        else:
            self.progress = tqdm.tqdm(total=self._total_size, unit="bytes", unit_scale=True, **tqdm_kwargs)
            self.is_progress_external = False

        super().__init__(path, *args, **kwargs)

    def detach_progress(self) -> tqdm.tqdm:
        """Detach and return progress object."""
        progress = self.progress
        self.progress = None
        self.is_progress_external = False
        return progress

    def read(self, size: int = -1) -> bytes:
        """Read up to size bytes from the object and return them.

        As a convenience, if size is unspecified or -1, all bytes until EOF are returned. Otherwise, only one system call is ever made.
        Fewer than size bytes may be returned if the operating system call returns fewer than size bytes.

        If 0 bytes are returned, and size was not 0, this indicates end of file. If the object is in non-blocking mode and no bytes are available,
        None is returned.

        Args:
        size (int): number of bytes to read
        """
        if self.progress:
            self.progress.update(size)
        return super().read(size)

    def close(self) -> None:
        """Flush and close this stream.

        This method has no effect if the file is already closed. Once the file is closed, any operation on the file
        (e.g. reading or writing) will raise a ValueError.

        As a convenience, it is allowed to call this method more than once; only the first call, however, will have an effect.
        """
        if self.progress and not self.is_progress_external:
            self.progress.close()
        return super().close()


def ensure_unpacked_sessions(raw_data_path: str, scratch_path: str, to_unpack: List[str]) -> None:
    """Ensure that all sessions in the raw data path are unpacked into the scratch path.

    Args:
        raw_data_path (str): Path to the raw data directory containing session files.
        scratch_path (str): Path to the scratch directory where sessions should be unpacked.
    """
    sessions = find_sessions_path(root_dir=raw_data_path)
    if any(s['is_compressed'] for s in sessions) is False:
        # print("No compressed sessions found in the raw data path. Nothing to unpack.")
        return
    else:
        logging.info("Found {} compressed sessions to unpack.".format(len([s for s in sessions if s['is_compressed']])))

    os.makedirs(scratch_path, exist_ok=True)
    for session in tqdm(sessions, desc='Unpacking sessions', leave=False):
        if session['is_compressed']:
            unpack_session(session, to_unpack, output_dir=scratch_path)