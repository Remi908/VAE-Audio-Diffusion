import json
import os
import random
from functools import partial
from glob import glob
from multiprocessing import Pool, cpu_count

import torch
import torchaudio
from torchaudio import transforms as T
import tqdm

from audio_diffusion.utils import Stereo, PadCrop, RandomPhaseInvert
from dataset.tempo_features import make_tempo_condition


class SampleDataset(torch.utils.data.Dataset):
    def __init__(self, paths, global_args):
        super().__init__()

        self.filenames = []

        print(f"Random crop: {global_args.random_crop}")

        self.augs = torch.nn.Sequential(
            PadCrop(global_args.sample_size, randomize=global_args.random_crop),
            RandomPhaseInvert(),
        )

        self.encoding = torch.nn.Sequential(
            Stereo()
        )

        for path in paths:
            for ext in ["wav", "flac", "ogg", "aiff", "aif", "mp3"]:
                self.filenames += glob(
                    f"{path}/**/*.{ext}",
                    recursive=True,
                )

        self.sr = global_args.sample_rate
        self.num_gpus = global_args.num_gpus
        self.cache_training_data = global_args.cache_training_data

        # self.use_tempo_conditioning = getattr(
        #     global_args,
        #     "use_tempo_conditioning",
        #     False,
        # )
        self.use_tempo_conditioning = (
            getattr(global_args, "use_tempo_conditioning", False)
            or getattr(global_args, "latent_dim", 0) > 0
        )

        self.min_bpm = getattr(
            global_args,
            "min_bpm",
            60.0,
        )

        self.max_bpm = getattr(
            global_args,
            "max_bpm",
            180.0,
        )

        self.tempo_cache = None

        if self.use_tempo_conditioning:
            tempo_cache_path = getattr(
                global_args,
                "tempo_cache_path",
                "tempo_cache.json",
            )

            with open(tempo_cache_path, "r") as f:
                self.tempo_cache = json.load(f)

            print(f"Loaded tempo cache: {tempo_cache_path}")

        if hasattr(global_args, "load_frac"):
            self.load_frac = global_args.load_frac
        else:
            self.load_frac = 1.0

        if self.cache_training_data:
            self.preload_files()

    def load_file(self, filename):
        audio, sr = torchaudio.load(filename)

        if sr != self.sr:
            resample_tf = T.Resample(sr, self.sr)
            audio = resample_tf(audio)

        return audio

    def load_file_ind(self, file_list, i):
        return self.load_file(file_list[i]).cpu()

    def get_data_range(self):
        start, stop = 0, len(self.filenames)

        try:
            local_rank = int(os.environ["LOCAL_RANK"])
            world_size = int(os.environ["WORLD_SIZE"])

            interval = stop // world_size

            start = local_rank * interval
            stop = (local_rank + 1) * interval

            print(
                "local_rank, world_size, start, stop =",
                local_rank,
                world_size,
                start,
                stop,
            )

            return start, stop

        except KeyError:
            start = 0
            stop = len(self.filenames) // self.num_gpus
            return start, stop

    def preload_files(self):
        n = int(len(self.filenames) * self.load_frac)

        print(f"Caching {n} input audio files:")

        wrapper = partial(
            self.load_file_ind,
            self.filenames,
        )

        start, stop = self.get_data_range()

        with Pool(processes=cpu_count()) as p:
            self.audio_files = list(
                tqdm.tqdm(
                    p.imap(wrapper, range(start, stop)),
                    total=stop - start,
                )
            )

    def get_tempo_condition(self, audio_filename):
        if self.tempo_cache is None:
            raise RuntimeError(
                "Tempo conditioning is enabled, but tempo_cache was not loaded."
            )

        # Try exact path first
        bpm = self.tempo_cache.get(audio_filename)

        # Fallback: try resolved absolute path
        if bpm is None:
            bpm = self.tempo_cache.get(str(os.path.abspath(audio_filename)))

        if bpm is None:
            raise KeyError(
                f"No BPM found in tempo cache for: {audio_filename}"
            )

        cond = make_tempo_condition(
            bpm,
            min_bpm=self.min_bpm,
            max_bpm=self.max_bpm,
        )

        return torch.tensor(
            cond,
            dtype=torch.float32,
        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        audio_filename = self.filenames[idx]

        try:
            if self.cache_training_data:
                audio = self.audio_files[idx]
            else:
                audio = self.load_file(audio_filename)

            if self.augs is not None:
                audio = self.augs(audio)

            audio = audio.clamp(-1, 1)

            if self.encoding is not None:
                audio = self.encoding(audio)

            if self.use_tempo_conditioning:
                tempo_cond = self.get_tempo_condition(audio_filename)
                return audio, tempo_cond, audio_filename

            return audio, audio_filename

        except Exception:
            return self[random.randrange(len(self))]