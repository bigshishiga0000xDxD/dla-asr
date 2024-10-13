import sys
import hydra
from hydra.utils import instantiate
from tqdm import tqdm

import torch

sys.path.append(".")

from src.datasets.data_utils import get_dataloaders


@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(config):
    device = "cpu"

    text_encoder = instantiate(config.text_encoder)
    dataloaders, batch_transforms = get_dataloaders(config, text_encoder, device)

    dataloader = dataloaders["train"]

    assert config.dataloader.batch_size == 1, "We don't want no paddings"

    elems = []
    for batch in tqdm(dataloader):
        spectrogram = batch["spectrogram"].half().flatten()
        elems.append(spectrogram)

    elems = torch.cat(elems)

    print("mean:", torch.mean(elems))
    print("std:", torch.std(elems))


if __name__ == "__main__":
    main()
