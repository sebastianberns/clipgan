# CLIPGAN

Implementation by Sebastian Berns ([@sebastianberns](https://twitter.com/sebastianberns)), based on the [Big Sleep colab notebook](https://colab.research.google.com/drive/1NCceX2mbiKOSlAd_o7IU7nA9UskKN5WR?usp=sharing) by Ryan Murdock ([@advadnoun](https://twitter.com/advadnoun)).

Uses OpenAI’s [CLIP implementation](https://github.com/openai/CLIP) and a modified version of the [BigGAN for Pytorch implementation](https://github.com/huggingface/pytorch-pretrained-BigGAN) by huggingface.

Additional features:

- Load pre-trained CLIP and BigGAN models from path

## Dependencies

Install required packages:

```shell
pip install -r requirements.txt
```

General:

- Pytorch (tested on version 1.7.1 and CUDA 10.1)
- Torchvision (tested on version 0.8.2)
- tqdm

CLIP:

- ftfy
- Pillow
- regex

BigGAN:

- boto3
- requests


## Run

Arguments:

- `text_prompt` (str, required) Generated images will be optimized to match this text input
- `clip_name_or_path` (str) Name of pre-trained model to load or path to state dict (default: 'ViT-B/32')
- `generator` (str) Name of generator architecture (default: 'biggan')
- `g_name_or_path` (str) Name of pre-trained model to load or path to state dict (default: 'biggan-deep-512')
- `steps` (int) Number of optimization steps (default: 500)
- `batch_size` (int) Number of images to generate and optimize in parallel (default: 32)
- `lr` (float) Optimization step size for Adam optimizer (default: 0.07)
- `beta1` (float) and `--beta2` (float) Adam optimizer parameters (default: 0.9 and 0.999)
- `save_path` (str) Path to save directory (default: './save'). A new folder based on the text prompt will be created here. Be careful, files will be overwritten without warning!
- `seed` (int) Random seed number (default: current time)
- `device` (str) Device to run models on (default: 'cuda')

### From the command line

Pre-prend arguments with a double dash (except for `text_prompt`)

```python
python main.py "Text prompt" \
--clip_name_or_path './models/ViT-B-32.pt' \
--g_name_or_path './models/biggan-deep-512/'
--steps 1000
--save_path './output'
```

### From another script

```python
from clipgan load CLIPGAN

model = CLIPGAN("Text prompt",
                clip_name_or_path='./models/ViT-B-32.pt',
                g_name_or_path='./models/biggan-deep-512/',
                steps=1000,
                save_path='./output')
model.run()
```
