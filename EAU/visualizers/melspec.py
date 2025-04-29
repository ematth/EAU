#
# Visualizer STFT Module
#

# TODO: Add jit support for batch MelSpectrogram processing.
import torch.jit as jit
from torchaudio.transforms import MelSpectrogram

class MelSpectrogram:
    def __init__(self, sr: int, dft_size: int, hop_size: int, window: torch.hann_window, win_length: int, device: str = 'cpu'):
        self.sample_rate=sr,
        self.dft_size=dft_size,
        self.window=torch.hann_window,
        self.win_length=dft_size,
        self.hop_length=hop_size,
        self.f_min=0.0,
        self.f_max=None,
        self.n_mels=128,
        self.pad=0,
        self.power=2.0,
        self.normalized=False,
        self.center=True,
        self.pad_mode='reflect',
        self.device = device

    def _melspec(self, input: torch.Tensor, sr: int) -> torch.Tensor:
        """Compute the Mel Spectrogram of a single input tensor.
        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, channels, time).
        Returns:
            torch.Tensor: Mel Spectrogram of the input tensor.
        """

        match input.ndim:
            case 1:
                return MelSpectrogram(
                    sample_rate=self.sample_rate,
                    n_fft=self.dft_size,
                    window_fn=self.window,
                    win_length=self.dft_size,
                    hop_length=self.hop_size,
                    f_min=self.f_min,
                    f_max=self.f_max,
                    n_mels=self.n_mels,
                    pad=self.pad,
                    power=self.power,
                    normalized=self.normalized,
                    center=self.center,
                    pad_mode=self.pad_mode,
                    device=self.device
                )(input)
            case 2:
                return MelSpectrogram(
                    sample_rate=self.sample_rate,
                    n_fft=self.dft_size,
                    window_fn=self.window,
                    win_length=self.dft_size,
                    hop_length=self.hop_size,
                    f_min=self.f_min,
                    f_max=self.f_max,
                    n_mels=self.n_mels,
                    pad=self.pad,
                    power=self.power,
                    normalized=self.normalized,
                    center=self.center,
                    pad_mode=self.pad_mode,
                    device=self.device
                )(input.mean(dim=0, keepdim=False))
            case _:
                raise Exception('Input tensor must be 1D, (samples), 2D (channels, samples)')


    def __call__(self, input: torch.Tensor, sr: int) -> torch.Tensor:
        return self._melspec(self, input, sr)