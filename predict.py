import os
import cog
import tempfile
import zipfile
from pathlib import Path
from midi2audio import FluidSynth
from model import PopMusicTransformer
import tensorflow as tf


class Predictor(cog.Predictor):
    def setup(self):
        """Init midi generator and midi-to-wav model"""
        self.model = PopMusicTransformer(
            checkpoint="REMI-tempo-checkpoint", is_training=False
        )
        self.fs = FluidSynth()

    @cog.input(
        "target_bars", type=int, default=16, help="number of midi bars to generate"
    )
    @cog.input(
        "temperature", type=float, default=1.2, help="stochastic sampling temperature"
    )
    @cog.input(
        "topk", type=int, default=5, help="top k alternatives for stochastic sampling"
    )
    @cog.input("seed", type=int, default=-1, help="Random seed, -1 for random")
    def predict(self, target_bars, temperature, topk, seed):
        """Generate midi and wav files"""
        if seed < 0:
            seed = int.from_bytes(os.urandom(2), "big")
        tf.random.set_random_seed(seed)

        out_path = Path(tempfile.mkdtemp())
        out_path_midi = out_path / "output.midi"
        out_path_wav = out_path / "output.wav"

        # generate from scratch
        self.model.generate(
            n_target_bar=target_bars,
            temperature=temperature,
            topk=topk,
            output_path=str(out_path_midi),
            prompt=None,
        )
        #self.model.close()

        # compute soundfile from midi score
        self.fs.midi_to_audio(str(out_path_midi), str(out_path_wav))

        return out_path_wav
