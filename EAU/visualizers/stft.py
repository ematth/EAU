#
# Visualizer STFT Module
#
from torch.fft import rfft, irfft

# TODO: Add jit support for batch STFT processing.
import torch.jit as jit

class STFT:
    def __init__(self, dft_size: int, hop_size: int, zero_pad: int = 0, device: str = 'cpu'):
        self.dft_size = dft_size
        self.hop_size = hop_size
        self.zero_pad = zero_pad
        self.window = torch.hann_window(dft_size, periodic=True, device=device)
        self.device = device

    # TODO: replace with hand-written STFT code.
    def _stft(self, input: torch.Tensor) -> torch.Tensor | Exception:
        """Compute the Short-Time Fourier Transform (STFT) of a single input tensor.
        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, channels, time).
        Returns:
            torch.Tensor: STFT of the input tensor.
        """

        match input.ndim:
            case 1:
                return rfft(self.window * array([input[i:i + self.dft_size] for i in range(0, len(input) - self.dft_size, self.hop_size)]), dim=-1, n=self.dft_size + self.zero_pad).T
            case 2:
                pass
            case _:
                exception('Input tensor must be 1D, (samples), 2D (channels, samples), or 3D (batches, channels, samples).')


    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        return self._stft(self, input)


class iSTFT:
    def __init__(self, dft_size: int, hop_size: int, zero_pad: int = 0, device: str = 'cpu'):
        self.dft_size = dft_size
        self.hop_size = hop_size
        self.zero_pad = zero_pad
        self.device = device

    def _istft(self, input: torch.Tensor) -> torch.Tensor | Exception:
        # TODO: replace with hand-written iSTFT code.
        pass

    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        return self._istft(self, input)