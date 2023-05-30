from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import torchdata.datapipes.iter as dpiter
from utfrecord.async_reader import IoUringTfrecordReaderParsed
import numpy as np
from tqdm import tqdm
import nvidia.dali.tfrecord as tfrec
from utfrecord.utils import transpose


@pipeline_def
def simple_pipeline(dp):
    jpegs, labels = fn.external_source(
        dp, num_outputs=2,
        dtype=[types.UINT8, types.INT64],
        no_copy=True,
    )
    images = fn.decoders.image(jpegs, device='mixed', output_type=types.RGB)
    images = fn.resize(
        images,
        device='gpu',
        resize_shorter=256,
    )
    images = fn.crop_mirror_normalize(
        images.gpu(),
        dtype=types.FLOAT,
        output_layout='CHW',
        crop=(224, 224),
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        mirror=False,
    )
    return images, labels.gpu()


@pipeline_def
def file_pipeline(root: str):
    jpegs, labels = fn.readers.file(
        file_root=root,
        name='Reader'
    )
    images = fn.decoders.image(jpegs, device='mixed', output_type=types.RGB)
    images = fn.resize(
        images,
        device='gpu',
        resize_shorter=256,
    )
    images = fn.crop_mirror_normalize(
        images.gpu(),
        dtype=types.FLOAT,
        output_layout='CHW',
        crop=(224, 224),
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        mirror=False,
    )
    return images, labels.gpu()


@pipeline_def
def tfrecord_pipeline(root: str):
    rec_dp = dpiter.FileLister(root, '*.tfrecord')
    idx_dp = dpiter.FileLister(root, '*.tfrecord.idx')
    inputs = fn.readers.tfrecord(
        path=list(rec_dp),
        index_path=list(idx_dp),
        features=dict(
            image=tfrec.FixedLenFeature((), tfrec.string, ""),
            label=tfrec.FixedLenFeature([1], tfrec.int64, -1)
        ),
        name='Reader'
    )
    jpegs, labels = inputs["image"], inputs["label"]
    images = fn.decoders.image(jpegs, device='mixed', output_type=types.RGB)
    images = fn.resize(
        images,
        device='gpu',
        resize_shorter=256,
    )
    images = fn.crop_mirror_normalize(
        images.gpu(),
        dtype=types.FLOAT,
        output_layout='CHW',
        crop=(224, 224),
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        mirror=False,
    )
    return images, labels.gpu()


def main():
    batch_size = 128
    num_threads = 8
    device_id = 4

    dp = dpiter.FileLister(
        'imagenet-tfrec/train', masks='*.tfrecord'
    )
    dp = dpiter.ShardingFilter(dp)
    dp = IoUringTfrecordReaderParsed(
        dp, keys=['image', 'label'],
        queue_depth=32, channel_size=10240,
    )

    def _parse_record(record):
        image_bytes = record['image'][0].numpy()
        label = np.array(record['label'])
        return image_bytes, label

    dp = dpiter.Mapper(dp, fn=_parse_record)
    dp = dpiter.Batcher(dp, batch_size=batch_size)
    dp = dpiter.Mapper(dp, fn=transpose)

    pipe = simple_pipeline(dp, batch_size=batch_size,
                           device_id=device_id, num_threads=num_threads)
    pipe.build()
    loader = DALIGenericIterator(
        pipe, ['image', 'label'],
    )

    # pipe = file_pipeline(
    #     'imagenet/train',
    #     batch_size=batch_size,
    #     device_id=5, num_threads=8
    # )
    # pipe = tfrecord_pipeline(
    #     'imagenet-tfrec/train',
    #     batch_size=batch_size,
    #     device_id=device_id,
    #     num_threads=num_threads
    # )
    # pipe.build()
    # loader = DALIGenericIterator(
    #     pipe, ['image', 'label'],
    #     reader_name='Reader',
    # )

    with tqdm() as pbar:
        for batch in loader:
            image = batch[0]['image']
            # print(batch0['image'].device)
            # break
            pbar.update(image.size(0))


if __name__ == '__main__':
    main()
