import logging
import pathlib

from typing import List
from typing import Generator

import cv2
import numpy


def load_frames(paths: List[str]) -> Generator[numpy.ndarray, None, None]:
    for path in paths:
        path = pathlib.Path(path)
        if path.is_dir():
            yield from load_frames(path.rglob('*'))
        elif path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            yield cv2.imread(str(path))
        else:
            logging.warning(f'skipping {path.name}...')
