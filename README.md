# Automatic Speech Recognition (ASR) with PyTorch

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## Installation

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```

## How To Use

* download pretrained model with
   ```bash
   gdown "https://drive.google.com/uc?id=1cZDbad3YpXuaouZ6ZuyQwEDjf4C5veVF"
   ```

* train baseline model with
   ```bash
   python train.py
   ```
* infer model with
   ```bash
   python inference.py inferencer.save_path="your_save_dir" inferencer.from_pretrained="your_checkpoint.pth"
   ```
* to check handcrafted beam search and language model append options:
   ```bash
   metrics=inference_metrics_test text_decoder="\$\{text_decoders.beam_search_slow_decoder\}"
   ```

   `metrics` for all metrics for beam search/lm beam search, `text_decoder` for getting raw predictions from beam search decoder. ours SOTA decoder is `${text_decoders.beam_search_fast_lm_unigram_decoder}`(WARNING: SUPER SLOW!)
* to evaluate on your own dataset append options
   ```bash
   datasets=custom-dir datasets.test.audio_dir="your_dir_with_audio_files" datasets.test.transcription_dir="your_dir_with_.txt_files"
   ```

   `transcription_dir` can be omitted. audio files and transcription files should have matching names.
* to compute WER and CER on evaluated custom dataset, run `python utils/compute_metrics.py --path YOUR_PATH_DIR`

## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
