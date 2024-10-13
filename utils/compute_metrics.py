from pathlib import Path
import os
import sys
import argparse

import torch
import numpy as np

sys.path.append('.')

from src.metrics.utils import calc_cer, calc_wer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='Path to the predictions folder with .pth files')
    args = parser.parse_args()
    path = Path(args.path)

    cers, wers = [], []
    for file in os.listdir(path):
        file = path / file
        output = torch.load(open(file, 'rb'))
        label = output['label']
        pred_label = output['pred_label']

        cers.append(calc_cer(label, pred_label))
        wers.append(calc_wer(label, pred_label))
    
    print('CER:', np.mean(cers) * 100)
    print('WER:', np.mean(wers) * 100)


if __name__ == '__main__':
    main()