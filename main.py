#!/usr/bin/env python

# System packages
import argparse
from pathlib import Path
import random
from time import time

# Global packages
import numpy as np
import torch
from torch.nn.functional import interpolate
from torchvision.transforms import Normalize
from torchvision.utils import save_image
from tqdm import trange

# Local packages
if __name__ == '__main__':  # Run code as script
    import clip
    from biggan import BigGAN
    from version import __version__
else:  # Run code as package
    from . import clip
    from .biggan import BigGAN
    from .version import __version__


def main():
    print(f"CLIPGAN (v{__version__})")

    # Script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('text_prompt', type=str)
    parser.add_argument('--clip_name_or_path', type=str, default='ViT-B/32',
                        help="Name of pre-trained model to load or path to state dict")
    parser.add_argument('--generator', choices=['biggan'],
                        default='biggan', help="Generator architecture")
    parser.add_argument('--g_name_or_path', type=str, default='biggan-deep-512',
                        help="Name of pre-trained model to load or path to state dict")
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.07)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--save_path', type=path_to_dir, default='./save')
    parser.add_argument('--seed', type=int, default=int(time()))
    parser.add_argument('--deterministic', type=bool, default=False)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    print("Configuration:")
    for arg in vars(args):
        print(f"\t{arg}: {getattr(args, arg)}")
    print()

    # Pass argument namespace as dictionary
    # Arguments in the parser need to match
    # the arguments in the class constructor
    model = CLIPGAN(**vars(args))
    model.run()


def path_to_dir(path, makedir=True, makedir_parent=True):
    dir = Path(path).expanduser().resolve()
    if makedir and not dir.exists():
        dir.mkdir(parents=makedir_parent)
    if dir.exists() and not dir.is_dir():
        raise argparse.ArgumentTypeError(f"'{path}' is not a directory")
    return dir


def path_to_file(path):
    f = Path(path).expanduser().resolve()
    if not f.is_file():
        raise argparse.ArgumentTypeError(f"'{path}' is not a file")
    if not f.exists():
        raise argparse.ArgumentTypeError(f"'{path}' does not exist")
    return f


def set_random_seed(seed=int(time())):
    """ Set random global seed
        seed (int) (optional)
    """
    assert isinstance(seed, int)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_determinism(d=False):
    """ Make Pytorch and CUDA behave deterninistically as best as possible
        (no guarantees)
        https://pytorch.org/docs/stable/notes/randomness.html
    """
    assert isinstance(d, bool)

    torch.set_deterministic(d)
    torch.backends.cudnn.deterministic = d

    # Disabling the benchmarking feature with
    #   `torch.backends.cudnn.benchmark = False`
    # causes cuDNN to deterministically select an algorithm,
    # possibly at the cost of reduced performance.
    torch.backends.cudnn.benchmark = not d


def get_device(device='cuda'):
    """ Check for device availablity
        device (str)   Name of desired device
        Returns torch device object
    """
    if not torch.cuda.is_available():
        device = 'cpu'
    return torch.device(device)


class Latents(torch.nn.Module):
    """ Latent variables and corresponding class labels
        batch_size (int)    Number of samples (latent vectors)
        z_dim (int)         Number of latent variables in each sample
        num_classes (int)   Number of classes. Set to 0 for unconditional case
    """
    def __init__(self, batch_size=32, z_dim=128, num_classes=1000,
                 device='cuda'):
        super(Latents, self).__init__()
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.device = get_device(device)

        # Sample latent vectors from a standard Gaussian (mean 0, std dev 1)
        latents = torch.zeros(batch_size, z_dim, device=self.device).normal_(std=1)
        self.latents = torch.nn.Parameter(latents)

        # Sample class
        if num_classes > 0:
            classes = torch.zeros(batch_size, num_classes, device=self.device).normal_(-3.9, .3)
            # class_id = 117
            # classes = torch.zeros(batch_size, num_classes)
            # classes[:, class_id] = 1.0
            self.classes = torch.nn.Parameter(classes)

    def forward(self):
        if self.num_classes > 0:
            return self.latents, torch.sigmoid(self.classes)
        else:
            return self.latents


class CLIPGAN:
    """ CLIPGAN
        text_prompt (str, required) Generated images will be optimized to match
                                    this text input
        clip_name_or_path (str)     Name of pre-trained CLIP model or path to
                                    saved state dict
        generator (str)             Name of generator architecture
        g_name_or_path (str)        Name of pre-trained generator model or path
                                    to saved state dict
        steps (int)                 Number of optimization steps
        batch_size (int)            Number of latent vector samples
        lr (float)                  Optimization step size for Adam optimizer
        beta1, beta2 (float)        Adam optimizer parameters
        save_path (str)             Path to save directory. A new folder based
                                    on the text prompt will be created here.
                                    Be careful, files will be overwritten
                                    without warning!
        seed (int)                  Random seed number
        deterministic (bool)        Flag to make generation reproducible
        device (str)                Device to run models and optimization on
    """
    def __init__(self, text_prompt,
                 clip_name_or_path='ViT-B/32',  # CLIP model name or path
                 generator='biggan',  # Generator architecture
                 g_name_or_path='biggan-deep-512',  # Model name or path
                 steps=500,
                 batch_size=32,  # Number of latent vectors (input to BigGAN)
                 lr=0.07,
                 beta1=0.9,
                 beta2=0.999,
                 save_path='save/',
                 seed=int(time()),
                 deterministic=False,
                 device='cuda'):
        self.text_prompt = text_prompt
        text_prompt_dir = self.text_prompt.replace(" ", "_")
        self.steps = steps
        self.batch_size = batch_size
        self.lr = lr
        self.betas = (beta1, beta2)
        self.save_path = Path(save_path)/text_prompt_dir
        self.save_path.mkdir(parents=True, exist_ok=True)  # Create directory
        self.seed = seed
        self.deterministic = deterministic
        self.device = get_device(device)

        set_random_seed(self.seed)
        set_determinism(self.deterministic)

        # Load CLIP model and transform
        self.clip_model, clip_transform = clip.load(clip_name_or_path, device=self.device)

        # Load generator model
        self.generator = BigGAN.from_pretrained(g_name_or_path,
                                                cache_dir='./models/'
                                                ).to(self.device).eval()

        # Preprocess prompt and get its CLIP embedding
        text_tokens = clip.tokenize(self.text_prompt).to(self.device)
        self.text_embed = self.clip_model.encode_text(text_tokens).detach().clone()

        # Initialize latent vectors
        self.latents = Latents(self.batch_size, device=self.device)

        # Set up optimization scheme
        self.optimizer = torch.optim.Adam(self.latents.parameters(),
                                          lr=self.lr, betas=self.betas,
                                          eps=1e-08, weight_decay=0)

    def run(self):
        for step in trange(1, self.steps+1):
            best_img = self.optimize(step)
            self.save_images(best_img, step)

    def optimize(self, step):
        # Get latent variables and class labels
        latvecs, latcls = self.latents()

        # Generate BigGAN samples (truncation 1)
        with torch.no_grad():
            gen_imgs = self.generator(latvecs, latcls, 1)

        clip_input = self.clip_image_preprocessing(gen_imgs, num_cuts=128)
        # Compute CLIP image encoding
        gen_imgs_embed = self.clip_model.encode_image(clip_input)

        lat_loss = self.compute_latent_loss(latvecs, threshold=1)
        class_loss = self.compute_class_loss(latcls)
        sim_loss_txt_img = self.compute_sim_loss_txt_img(self.text_embed,
                                                         gen_imgs_embed)

        # Losses to be minimized
        # 1. Latent loss
        # 2. Class loss
        # 3. Similarity loss text-image
        # 100 is a weighting factor (gives more importance to this term)
        loss = lat_loss + class_loss + 100 * sim_loss_txt_img
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Pick index of best image (lowest class loss)
        best = torch.topk(class_loss, k=1, largest=False)[1]
        return gen_imgs[best:best+1]

    def clip_image_preprocessing(self, images, num_cuts=32):
        """ Preprocess generated images to be fed into CLIP
            Increase the number of images by cutting out random subimages
            Convert to Tensor
            Change range of values from [-1, +1] to [0, 1]
            Normalize by ImageNet statistics
        """
        # ImageNet means and std by channel
        normalize = Normalize((0.48145466, 0.4578275, 0.40821073),
                              (0.26862954, 0.26130258, 0.27577711))

        # Check dimensions of generated images
        _, _, width, height = images.size()
        assert width >= height

        clip_img_size = self.clip_model.input_resolution.item()

        cuts = []
        for _ in range(num_cuts):
            # Determine a smaller size to cut images out of the generated samples
            # 1. Draw one sample from a Gaussian (centered at 0.8 with std dev 0.3)
            # 2. Limit the value to the range [0.5, 0.95]
            # 3. Multiply original size with smaller factor
            # 4. Convert to integer
            size = int(width * torch.zeros(1,).normal_(.8, .3).clip(.5, .95))

            # Get random offsets from the edge of the image
            offsetx = torch.randint(0, width - size, ())
            offsety = torch.randint(0, width - size, ())

            # Cut out images from generated samples
            # Dimensions are [Batch, Channels, Width, Height]
            # Here only width and height are touched
            cut = images[:, :, offsetx:offsetx + size, offsety:offsety + size]

            # Scale random cuts to CLIP input size
            # torch.nn.functional.interpolate
            cut = interpolate(cut, (clip_img_size, clip_img_size), mode='nearest')

            cuts.append(cut)  # Collect cuts in list

        cuts = torch.cat(cuts, dim=0)  # Create tensor from list
        cuts = normalize((cuts + 1) / 2)  # Normalize
        return cuts

    def compute_latent_loss(self, latvecs, threshold=1):
        """ Latent vector loss
            Force distribution of values in latent vectors to be as close as
            possible to a standard Gaussian
        """
        threshold = torch.tensor(threshold, device=latvecs.device)

        # First and second standardized moments (mean and standard deviation)
        mean = latvecs.mean().abs()  # Absolute mean of latents
        # Mean of absolute difference of lat vec’s stdev to 1 (normal Gaussian)
        std = torch.abs(1 - latvecs.std(dim=1)).mean()
        max = torch.max(latvecs.square().mean(),  # Mean of squared latent vecs
                        threshold)  # Latent threshold (default: 1)

        lat_loss = mean + std + 4 * max

        # Score normalization for calculation of skewness and kurtosis
        # In batch but for each latent vector (each row) separately
        vec_means = latvecs.mean(dim=1, keepdim=True)  # Compute means
        vec_diffs = latvecs - vec_means  # Compute deviations from means
        vec_vars = vec_diffs.square().mean(dim=1, keepdim=True)  # Variances
        vec_stds = vec_vars.sqrt()  # Standard deviations
        vec_scores = vec_diffs / vec_stds  # Normalized scores

        # Third and fourth standardized moments
        # Compute skewness and kurtosis of values in latent vector
        skewness = vec_scores.pow(3.0).mean(dim=1)
        kurtosis = vec_scores.pow(4.0).mean(dim=1) - 3.0

        # Absolute deviations from normal Gaussian skewness (0) and kurtosis (3)
        # Compute average over all vectors (rows)
        skewness = skewness.abs().mean()
        kurtosis = kurtosis.abs().mean()

        # Add to loss
        lat_loss += skewness + kurtosis

        return lat_loss

    def compute_class_loss(self, latcls):
        """ Class loss """
        k = latcls.size(1)  # Get number of classes

        # 1. Find 999 lowest class values, i.e. leave out the highest class value
        topk_values, _ = torch.topk(latcls, k=k-1, dim=1, largest=False)
        # 2. Multiply by 50 -> any value over or equal to 0.02 will be higher or equal to 1
        topk_values *= 50
        # 3. Square -> any value below 1 will decrease, any value above 1 (prev min 0.02) will increase
        # 4. Take the mean
        return topk_values.square().mean()

    def compute_sim_loss_txt_img(self, text_embed, images_embed):
        """ Similarity loss comparing embedded text prompt to embedded images
            Needs to be negative because we want to increase similarity
        """
        sim = torch.cosine_similarity(text_embed, images_embed, dim=-1)
        return -sim.mean()

    def save_images(self, image, step):
        pad = len(str(self.steps))  # Number of digits for padding
        filename = f"step_{step:0{pad}}.png"  # Prefix filename with zeros
        save_image(image, self.save_path/filename,  # torchvision method
                   normalize=True, range=(-1., +1.))


if __name__ == '__main__':
    main()
