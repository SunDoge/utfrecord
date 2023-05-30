import torchdata.datapipes.iter as dpiter
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utfrecord.sync_reader import TfrecordReader, TfrecordReaderParsed
from utfrecord.async_reader import IoUringTfrecordReaderParsed
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

USE_UTFRECORD = True

dp = dpiter.FileLister('imagenet-tfrec/train', masks='*.tfrecord')
dp = dpiter.ShardingFilter(dp)

# print(list(dp))

spec = dict(
    image=(tuple(), None),
    label=(tuple(), torch.int64)
)

if USE_UTFRECORD:
    # dp = TfrecordReader(dp, channel_size=1024,
    #                     spec=spec, check_integrity=False)
    # dp = TfrecordReaderParsed(dp, ['image', 'label'], channel_size=102400, check_integrity=False)
    dp = IoUringTfrecordReaderParsed(
        dp, keys=['image', 'label'], queue_depth=8, channel_size=102400, check_integrity=True
    )
else:
    dp = dpiter.FileOpener(dp, mode='rb')
    dp = dpiter.TFRecordLoader(dp, spec=spec)


aug = A.Compose([
    # A.SmallestMaxSize(max_size=256),
    # A.CenterCrop(224, 224),
    # A.Normalize(),
    A.Resize(112, 112, interpolation=cv2.INTER_NEAREST),
    ToTensorV2()
])


def parse_v1(example):
    image = example['image']
    label = example['label']
    img_buf = np.frombuffer(image, dtype=np.uint8)
    img = cv2.imdecode(img_buf, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = aug(image=img)['image']
    return dict(image=img_tensor, label=label)


def parse_v2(example):
    # print(example)
    image = example['image']
    label = example['label']
    img_buf = image[0].numpy()
    img = cv2.imdecode(img_buf, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = aug(image=img)['image']
    return dict(image=img_tensor, label=label)


# dp = dpiter.Mapper(dp, fn=parse_v2)


loader = DataLoader(dp, batch_size=256, num_workers=16,
                    worker_init_fn=lambda w: cv2.setNumThreads(0), collate_fn=lambda x: x)

with tqdm() as pbar:
    # for batch in loader:
    #     # pbar.update(batch['label'].size(0))
    #     pbar.update(len(batch))

    for data in dp:
        pbar.update()
