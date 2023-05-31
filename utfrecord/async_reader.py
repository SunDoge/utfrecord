
from torchdata.datapipes.iter import IterDataPipe
from typing import Optional, Iterable, List, Dict, Union, Iterator
from torch import Tensor
from . import utfrecord as _utfrecord
import warnings
from .utils import features_to_tensor


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
        self.paths = list(datapipe)

    def __iter__(self) -> Iterator[Dict[str, Union[Tensor, List[Tensor]]]]:

        reader = _utfrecord.IoUringTfrecordReader(
            self.paths, self.use_cycle, self.queue_depth, self.channel_size, self.keys, self.check_integrity
        )

        try:
            for example_dlpack in reader:
                example = features_to_tensor(example_dlpack)
                yield example

        except RuntimeError as e:
            warnings.warn(
                f"Unable to read from corrupted tfrecord stream {self.paths} due to: {e}, abort!")
            raise e

    def __len__(self) -> int:
        if self.length == -1:
            raise TypeError(
                f"{type(self).__name__} instance doesn't have valid length")
        return self.length
