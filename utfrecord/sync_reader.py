from typing import Iterator
from . import utfrecord as _utfrecord
from torchdata.datapipes.iter import IterDataPipe
from typing import Optional, Iterable, List, Dict, Union
from torchdata.datapipes.iter.util.tfrecordloader import TFRecordExampleSpec, TFRecordExample, parse_tfrecord_sequence_example
import warnings
from torch.utils.dlpack import from_dlpack
from torch import Tensor


class TfrecordReader(IterDataPipe):

    def __init__(
        self,
        datapipe: Iterable[str],
        spec: Optional[TFRecordExampleSpec] = None,
        channel_size: int = 1024,
        check_integrity: bool = False,
        length: int = -1,
    ) -> None:
        super().__init__()
        self.datapipe = datapipe
        self.spec = spec
        self.length = length
        self._channel_size = channel_size
        self._check_integrity = check_integrity

    def __iter__(self) -> Iterator[TFRecordExample]:
        from torchdata.datapipes.iter.util.protobuf_template import _tfrecord_example_pb2 as example_pb2

        for path in self.datapipe:
            reader = _utfrecord.KanalReceiver(
                path, self._channel_size, self._check_integrity
            )

            try:
                for example_dlpack in reader:
                    example_byte_tensor = from_dlpack(example_dlpack)
                    example = example_pb2.SequenceExample()
                    example.ParseFromString(example_byte_tensor.numpy())
                    yield parse_tfrecord_sequence_example(example, self.spec)

            except RuntimeError as e:
                warnings.warn(
                    f"Unable to read from corrupted tfrecord stream {path} due to: {e}, abort!")
                raise e

    def __len__(self) -> int:
        if self.length == -1:
            raise TypeError(
                f"{type(self).__name__} instance doesn't have valid length")
        return self.length


class TfrecordReaderParsed(IterDataPipe):

    def __init__(
        self,
        datapipe: Iterable[str],
        keys: List[str],
        use_cycle: bool = False,
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

    def __iter__(self) -> Iterator[Dict[str, Union[Tensor, List[Tensor]]]]:

        paths = list(self.datapipe)

        reader = _utfrecord.KanalReceiverParsed(
            paths, self.use_cycle, self.channel_size, self.keys, self.check_integrity
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
                f"Unable to read from corrupted tfrecord stream {path} due to: {e}, abort!")
            raise e

    def __len__(self) -> int:
        if self.length == -1:
            raise TypeError(
                f"{type(self).__name__} instance doesn't have valid length")
        return self.length
