import os
import numpy as np
from typing import List, Optional
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.editor import concatenate_videoclips
from pytube import __main__

from .feature import Feature
from .scenedet import SceneDetector


class Summarizer:
    def __init__(self, duration: int):
        self.duration = duration
        self.F = None
        self.S = None
        self.__features = None
        self.__timecodes = None
        self.__durations = None
        self.__final_n_frames = None
        self.__selected_scenes = None
        self.__outpath = None

    def set_feature_extractor(self, extractor: Feature):
        self.F = extractor
        self.__outpath = os.path.join(os.path.dirname(self.F.video_filepath), 'summary.mp4')

    def __set_scene_detector(self):
        self.S = SceneDetector(self.F.fps, self.F.n_frames)

    def __get_n_frames(self):
        fps = self.F.fps
        total_frames = self.F.n_frames

        ratio = (self.duration * fps) / (total_frames * fps)
        self.__final_n_frames = int(ratio * total_frames)

    def __fetch_features(self):
        self.F.extract()
        self.__features = self.F.get_features()

    def __detect_scenes(self):
        f = self.__features.copy()
        self.__timecodes, self.__durations, self.__features = self.S.detect_scenes(f)

    def summarize(self):
        print('Fetching features...')
        self.__fetch_features()
        print('Features fetched...')
        self.__get_n_frames()
        print('Getting `n` frames...')
        self.__set_scene_detector()
        print('Scene detector set; detecting scenes ...')
        self.__detect_scenes()
        print('Scenes detected, selecting scenes ...')
        selected_scenes = self.__knapsack()
        print('Scenes selected, generating summary ...')
        original = VideoFileClip(self.F.video_filepath)
        clips = []
        for i in range(1, len(selected_scenes)):
            start = self.__timecodes[selected_scenes[i - 1]]
            end = self.__timecodes[selected_scenes[i - 1] + 1]
            clip = original.subclip(start, end)
            clips.append(clip)

        summ = concatenate_videoclips(clips)
        summ.write_videofile(self.__outpath)
        return self.__outpath,"Success"

    def __knapsack(self):
        W = self.__final_n_frames
        w = self.__durations
        v = self.__features
        n = len(self.__durations)

        dp = [[0 for _ in range(W + 1)] for _ in range(n + 1)]
        dp = np.array(dp)

        for i in range(n + 1):
            for j in range(W + 1):
                if i == 0 or j == 0:
                    continue

                elif w[i - 1] <= j:
                    if_chosen = int(v[i - 1]) + dp[i - 1][j - w[i - 1]]
                    not_chosen = dp[i - 1][j]

                    dp[i][j] = max(if_chosen, not_chosen)
                else:
                    dp[i][j] = dp[i - 1][j]

        return self.__fetch_scenes(dp)

    def __fetch_scenes(self, dp: List[List[int]]):
        n = len(self.__durations)
        W = self.__final_n_frames
        v = self.__features
        res = dp[n][W]
        curr = W
        w = self.__durations

        selected = list()
        
        for i in range(n, 0, -1):
            if res <= 0:
                break

            if res == dp[i - 1][curr]:
                continue
            else:
                selected.append(i - 1)
                res = res - int(v[i - 1])
                curr = curr - w[i - 1]

        selected_scenes = sorted(list(set(selected)))
        return selected_scenes

# if __name__ == "__main__":
#     from multiprocessing import freeze_support
#     freeze_support()
#     f = Feature('cricket1.mp4')
#     summarizer = Summarizer(duration=120)
#     summarizer.set_feature_extractor(f)
#     summarizer.summarize()