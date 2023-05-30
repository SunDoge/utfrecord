from typing import List


def transpose(batch: List[List]) -> List[List]:
    return list(map(list, zip(*batch)))
