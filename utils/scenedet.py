import cv2
import numpy as np
from tqdm.auto import trange

class SceneDetector:

    def __init__(self, fps: int, n_frames: int, threshold: float = 0.45, verbose: bool = False):
        self.verbose = verbose
        self.threshold = threshold
        self.fps = fps
        self.n_frames = n_frames

    def detect_scenes(self, features: np.ndarray):
        features = features.astype('float32')
        n_frames, _ = features.shape
        correlations = np.zeros((n_frames, 1))
        iterable_ = range(1, n_frames)
        if self.verbose:
            iterable_ = trange(n_frames)
        for i in iterable_:
            correlations[i] = cv2.compareHist(features[i - 1], features[i], cv2.HISTCMP_CORREL)

        timecodes, duration = self.__get_timecodes(correlations)
        scene_features = self.__get_scene_features(features, correlations)
        return timecodes, duration, scene_features

    def __get_scene_features(self, features: np.ndarray, correlations: np.ndarray):
        idx = np.argwhere(correlations < self.threshold)
        print(idx)
        prev = 0
        _features = []
        for frame in idx:
            _x = np.sum(features[prev:frame[0]])
            _features.append(_x)
            prev = frame[0]
        return _features

    @staticmethod
    def convert_to_timecode(frame, fps):
        hours = int(frame / (3600 * fps))
        minutes = int(frame / (60 * fps) % 60)
        seconds = int(frame / fps % 60)
        frames = int(frame % fps)
        return '{:02}:{:02}:{:02}.{:02}'.format(hours, minutes, seconds, frames)

    def __get_timecodes(self, correlations):
        idx = np.argwhere(correlations < self.threshold)
        timecodes = []
        durations = []
        
        prev = 0

        for frame, _ in idx:
            timecodes.append(SceneDetector.convert_to_timecode(frame, self.fps))
            durations.append(frame - prev)
            prev = frame

        return timecodes, durations