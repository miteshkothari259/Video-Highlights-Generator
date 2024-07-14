import os
from math import ceil
from multiprocessing import Pool, RLock, cpu_count
from typing import List, Union

import cv2
import numpy as np
from tqdm.auto import tqdm


class Feature:

    COLOR_SPACES = {
        "rgb": {"channels": [0, 1, 2], "range": [0, 256] * 3},
        "hsv": {"channels": [0, 1], "range": [0, 180, 0, 256]},
        "gray": {"channels": [0], "range": [0, 256]},
    }

    def __init__(
        self,
        video_filepath: str,
        color_space: str = "hsv",
        bins_per_channel: Union[int, List] = [18, 8],
        scale_factor: int = 50,
        verbose: bool = False,
    ):
        self.video_filepath = video_filepath
        self.color_space = None
        self.bins_per_channel = None
        self.scale_factor = scale_factor
        self.verbose = verbose

        self.n_frames = 0
        self.fps = 0
        self.width = 0
        self.height = 0

        self.__jump_unit = 0
        self.__out_file = os.path.join(
            os.path.dirname(self.video_filepath), "features.npy"
        )
        self.__cleared = False
        self.__temp_file_format = os.path.join(
            os.path.dirname(self.video_filepath), ".temp_{}.npy"
        )

        if color_space not in self.COLOR_SPACES.keys():
            raise ValueError(
                f"Color spaces supported are `{self.COLOR_SPACES.keys()}`, but `{color_space}` was provided."
            )

        self.color_space = color_space
        self.__validate_bins(bins_per_channel)

    def __validate_bins(self, bins_per_channel: Union[int, List]):

        if isinstance(bins_per_channel, int):
            if self.color_space == "rgb":
                self.bins_per_channel = [bins_per_channel] * 3
            elif self.color_space == "gray":
                self.bins_per_channel = [bins_per_channel]
            else:
                self.bins_per_channel = [bins_per_channel] * 2

        elif isinstance(bins_per_channel, list):
            length = len(bins_per_channel)
            if self.color_space == "rgb":
                if length != 3:
                    raise ValueError(
                        f"RGB color space requires 3 values for each color channel, provided had `{length}` values."
                    )
                for n_bins in bins_per_channel:
                    if n_bins < 0 or n_bins > 256:
                        raise ValueError(
                            f"RGB color space has range [0 - 255], that is, 256 values per channel. Provided has value `{n_bins}`."
                        )
            elif self.color_space == "hsv":
                if length != 2:
                    raise ValueError(
                        f"HSV color space works on 2 channels `hue` and `saturation` for features. Provided length was `{length}`."
                    )
                for idx, n_bins in enumerate(bins_per_channel):
                    if idx == 0:
                        if n_bins < 0 or n_bins > 180:
                            raise ValueError(
                                f"Hue has range [0 - 179], that is, 180 values. Provided has value `{n_bins}`."
                            )
                    else:
                        if n_bins < 0 or n_bins > 256:
                            raise ValueError(
                                f"Saturation has range [0 - 255], that is, 256 values. Provided has value `{n_bins}`."
                            )
            else:
                if length != 1:
                    raise ValueError(
                        f"Grayscale image have only one channel, expected length of argument is 1 but `{length}` was provided."
                    )
                if bins_per_channel[0] < 0 or bins_per_channel[0] > 256:
                    raise ValueError(
                        f"Grayscale values have range [0 - 255], that is, 256 values. Provided has value `{bins_per_channel[0]}`."
                    )

            self.bins_per_channel = bins_per_channel

    def __get_video_details(self) -> None:
        cap = cv2.VideoCapture(self.video_filepath)
        self.n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

    def _batch_process(self, group_number: int) -> None:
        cap = cv2.VideoCapture(self.video_filepath)

        cap.set(cv2.CAP_PROP_POS_FRAMES, self.__jump_unit * group_number)

        new_width = int(self.width * self.scale_factor / 100)
        new_height = int(self.height * self.scale_factor / 100)

        new_dim = (new_width, new_height)

        histograms = []
        color_space_conversion = {
            "rgb": cv2.COLOR_BGR2RGB,
            "hsv": cv2.COLOR_BGR2HSV,
            "gray": cv2.COLOR_BGR2GRAY,
        }

        progress_bar = None
        if self.verbose:
            progress_bar = tqdm(total=self.__jump_unit)

        proc_frames = 0

        try:
            while proc_frames < self.__jump_unit:
                grabbed, frame = cap.read()

                if not grabbed:
                    break

                frame = cv2.resize(frame, new_dim)
                frame = cv2.cvtColor(frame, color_space_conversion[self.color_space])
                channels = Feature.COLOR_SPACES.get(self.color_space).get("channels")
                ranges = Feature.COLOR_SPACES.get(self.color_space).get("range")
                hist = cv2.calcHist(
                    [frame], channels, None, self.bins_per_channel, ranges
                )
                hist = cv2.normalize(hist, hist)
                hist = hist.flatten()
                # hist = cv2.normalize(hist, hist, dtype=cv2.CV_32F).flatten()
                histograms.append(hist)

                proc_frames += 1

                if self.verbose:
                    progress_bar.update()

        except cv2.error as e:
            print(f"[ERROR] {e}")
        finally:
            cap.release()

        hists = np.array(histograms)
        np.save(self.__temp_file_format.format(group_number), hists)

    def extract(self) -> None:
        self.__n_cores = cpu_count()
        self.__get_video_details()

        self.__jump_unit = ceil(self.n_frames / self.__n_cores)

        pool = Pool(processes=self.__n_cores)
        if self.verbose:
            tqdm.set_lock(RLock())
            pool = Pool(
                processes=self.__n_cores,
                initializer=tqdm.set_lock,
                initargs=(tqdm.get_lock(),),
            )

        pool.map(self._batch_process, range(self.__n_cores))

        self.__combine()

    def __combine(self) -> None:
        h = np.load(self.__temp_file_format.format(0))
        _, n_features = h.shape

        features = np.zeros((self.n_frames, n_features))

        for idx, proc in enumerate(range(self.__n_cores)):
            file = self.__temp_file_format.format(proc)
            histogram = np.load(file)
            start = idx * self.__jump_unit
            end = min(
                start + self.__jump_unit, self.n_frames, start + histogram.shape[0]
            )
            features[start:end, :] = histogram
            os.remove(file)

        np.save(self.__out_file, features)

    def get_features(self) -> np.ndarray:
        if not self.__cleared:
            return np.load(self.__out_file)
        raise AttributeError(
            "`Features` object was cleared before fetching the features."
        )

    def clear(self) -> None:
        os.remove(self.__out_file)
        self.__cleared = True