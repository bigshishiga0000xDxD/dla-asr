from torch import nn

from src.model.base import BaseModel

class DeepSpeech2Model(BaseModel):
    def __init__(
            self,
            in_channels: int,
            conv_type: type[nn.Conv1d] | type[nn.Conv2d],
            convs_channels: list[int],
            convs_kernels: list[int | tuple[int, int]],
            convs_strides: list[int | tuple[int, int]],
            rnn_type: type[nn.RNN] | type[nn.LSTM] | type[nn.GRU],
            n_rnn: int,
            hidden_size: int,
            n_tokens: int
        ):
        """
        DeepSpeech2 architechture (http://proceedings.mlr.press/v48/amodei16.pdf)

        Args:
            in_channels (int): number of input_channels (n_mels in spectrogram).
            conv_type (type[nn.Conv1d] | type[nn.Conv2d]): type of convolutions
                in pre-rnn layers (1d or 2d)
            convs_channels (list[int]): numbers of channels in convolutions
            convs_kernels (list[int | tuple[int, int]]): kernel_sizes in convolutions.
                Could be int or tuple[int, int] is case of 2d convolutions.
            convs_strides (list[int | tuple[int, int]]): strides in convolutions.
                Could be int or tuple[int, int] is case of 2d convolutions.
            rnn_type (type[nn.RNN] | type[nn.LSTM] | type[nn.GRU]): architechture
                used for rnn layers
            n_rnn (int): number of consecutive rnn layers
            hidden_size (int): hidden size of rnn layer. Should equal to `convs_channels[-1]`
                in case of 1d convolutions and `convs_channels[-1] * d` in case of 2d, where `d` is
                frequency dimension after all convolutions
            n_tokens (int): number of token in vocab
        """
        super().__init__()

        assert len(convs_channels) == len(convs_kernels) == len(convs_strides)

        self.convs = nn.ModuleList([
            conv_type(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=0,
                bias=False
            )
            for in_channels, out_channels, kernel_size, stride in zip(
                [in_channels] + convs_channels[:-1],
                convs_channels,
                convs_kernels,
                convs_strides
            )
        ])

        if conv_type == nn.Conv1d:
            batch_norm = nn.BatchNorm1d
        elif conv_type == nn.Conv2d:
            batch_norm = nn.BatchNorm2d

        self.conv_norms = nn.ModuleList([
            batch_norm(out_channels)
            for out_channels in convs_channels
        ])

        self.rnn = rnn_type(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_rnn,
            batch_first=True,
            bidirectional=False
        )

        self.debedder = nn.Linear(hidden_size, n_tokens)
    
    def forward(self, spectrogram, spectrogram_length, **batch):
        output = spectrogram
        if isinstance(self.convs[0], nn.Conv2d):
            # For 2d convolutions, we should have a channels dimension
            output = output.unsqueeze(1)

        for conv, norm in zip(self.convs, self.conv_norms):
            output = conv(output)
            output = norm(output)
        
        if len(output.shape) == 4:
            # For 2d convolutions, flatten spacial dimension afterwards
            b, c, h, l = output.shape
            output = output.reshape(b, c * h, l)
        
        output, _ = self.rnn(output.transpose(1, 2))
        output = self.debedder(output)

        log_probs = nn.functional.log_softmax(output, dim=-1)
        log_probs_length = self.transform_input_lengths(spectrogram_length)

        return {"log_probs": log_probs, "log_probs_length": log_probs_length}
    
    def transform_input_lengths(self, input_lengths):
        output_lengths = input_lengths
        for conv in self.convs:
            output_lengths = (output_lengths - conv.kernel_size[-1]) // conv.stride[-1] + 1
        return output_lengths