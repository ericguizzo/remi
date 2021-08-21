import os
import cog
import tempfile
import zipfile
from pathlib import Path
from midi2audio import FluidSynth
from model import PopMusicTransformer


class generateMIDI(cog.Predictor):
    def setup(self):
        '''Init midi generator and midi-to-wav model'''
        self.model = PopMusicTransformer(checkpoint='REMI-tempo-checkpoint',
                                        is_training=False)
        self.fs = FluidSynth()

    @cog.input("target_bars", type=int, default=16,
    help="number of midi bars to generate")
    @cog.input("temperature", type=float, default=1.2,
    help="stochastic sampling temperature")
    @cog.input("topk", type=int, default=5,
    help="top k alternatives for stochastic sampling")

    def predict(self, target_bars, temperature, topk):
        """Generate midi and wav files"""

        out_path = Path(tempfile.mkdtemp())
        zip_path = Path(tempfile.mkdtemp()) / "output.zip"
        out_path_midi = out_path / "output.midi"
        out_path_wav = out_path / "output.wav"

        # generate from scratch
        self.model.generate(
            n_target_bar=target_bars,
            temperature=temperature,
            topk=topk,
            output_path=str(out_path_midi),
            prompt=None)
        self.model.close()

        #compute soundfile from midi score
        self.fs.midi_to_audio(str(out_path_midi), str(out_path_wav))

        with zipfile.ZipFile(str(zip_path), "w") as zf:
            zf.write(str(out_path_midi))
            zf.write(str(out_path_wav))

        return zip_path
