from typing import List, Dict, Union, Any
from torch.utils.dlpack import from_dlpack
from torch import Tensor


def transpose(batch: List[List]) -> List[List]:
    return list(map(list, zip(*batch)))


def features_to_tensor(features: Dict[str, Union[List[Any], Any]]) -> Dict[str, Union[List[Tensor], Tensor]]:
    example = {}
    for key, feat in features.items():
        if isinstance(feat, list):
            example[key] = [from_dlpack(f) for f in feat]
        else:
            example[key] = from_dlpack(feat)
    return example
