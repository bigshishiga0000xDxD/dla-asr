import torch
from torch.utils.data import default_collate


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result = default_collate([
        {
            'spectrogram_length': item['spectrogram'].shape[-1],
            'text_encoded_length': item['text_encoded'].shape[-1],
            'text': item['text'],
            'audio_path': item['audio_path']
        }
        for item in dataset_items
    ])

    result.update({
        'spectrogram': torch.nn.utils.rnn.pad_sequence([
            item['spectrogram'].squeeze(0).T
            for item in dataset_items
        ], batch_first=True).transpose(1, 2),
        'text_encoded': torch.nn.utils.rnn.pad_sequence([
            item['text_encoded'].squeeze(0)
            for item in dataset_items
        ], batch_first=True),
        'audio': torch.nn.utils.rnn.pad_sequence([
            item['audio'].squeeze(0)
            for item in dataset_items
        ], batch_first=True)
    })

    return result
