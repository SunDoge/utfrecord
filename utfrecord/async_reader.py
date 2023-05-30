
from torchdata.datapipes.iter import IterDataPipe
from typing import Optional, Iterable, List, Dict, Union, Iterator
from torch import Tensor
from . import utfrecord as _utfrecord
from torch.utils.dlpack import from_dlpack
import warnings


class IoUringTfrecordReaderParsed(IterDataPipe):

    def __init__(
        self,
        datapipe: Iterable[str],
        keys: List[str],
        use_cycle: bool = False,
        queue_depth: int = 16,
        channel_size: int = 1024,
        check_integrity: bool = False,
        length: int = -1,
    ) -> None:
        super().__init__()
        self.datapipe = datapipe
        self.keys = keys
        self.length = length
        self.use_cycle = use_cycle
        self.channel_size = channel_size
        self.check_integrity = check_integrity
        self.queue_depth = queue_depth

    def __iter__(self) -> Iterator[Dict[str, Union[Tensor, List[Tensor]]]]:

        paths = list(self.datapipe)

        reader = _utfrecord.IoUringTfrecordReader(
            paths, self.use_cycle, self.queue_depth, self.channel_size, self.keys, self.check_integrity
        )

        try:
            for example_dlpack in reader:
                # example = {from_dlpack(example_dlpack[k]) for k in self.keys}
                example = {}
                for key in self.keys:
                    feat = example_dlpack[key]
                    if isinstance(feat, list):
                        example[key] = [from_dlpack(f) for f in feat]
                    else:
                        example[key] = from_dlpack(feat)
                yield example

        except RuntimeError as e:
            warnings.warn(
                f"Unable to read from corrupted tfrecord stream {paths} due to: {e}, abort!")
            raise e

    def __len__(self) -> int:
        if self.length == -1:
            raise TypeError(
                f"{type(self).__name__} instance doesn't have valid length")
        return self.length
