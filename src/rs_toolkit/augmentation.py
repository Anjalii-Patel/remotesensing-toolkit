"""Spatial data augmentation for training deep learning models on remote sensing patches."""
import numpy as np

def random_flip(patch, seed=None):
    """Randomly flip [C,H,W] patch horizontally and/or vertically."""
    rng = np.random.default_rng(seed)
    if rng.random() > 0.5: patch = patch[:, :, ::-1]  # horizontal
    if rng.random() > 0.5: patch = patch[:, ::-1, :]   # vertical
    return np.ascontiguousarray(patch)

def random_rot90(patch, seed=None):
    """Randomly rotate [C,H,W] patch by 0/90/180/270 degrees."""
    rng = np.random.default_rng(seed)
    k = rng.integers(0, 4)
    return np.ascontiguousarray(np.rot90(patch, k, axes=(1, 2)))

def add_gaussian_noise(patch, sigma=0.02, seed=None):
    """Add Gaussian noise to [C,H,W] patch."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, patch.shape).astype(patch.dtype)
    return patch + noise

def random_brightness(patch, max_delta=0.1, seed=None):
    """Shift brightness by a random uniform delta."""
    rng = np.random.default_rng(seed)
    delta = rng.uniform(-max_delta, max_delta)
    return patch + delta

def augment_batch(patches, n_aug=3, seed=0):
    """Augment a batch [N,C,H,W] by applying random transforms, returning N*(1+n_aug) patches."""
    rng = np.random.default_rng(seed)
    out = [patches]
    for _ in range(n_aug):
        aug = np.stack([
            add_gaussian_noise(random_flip(random_rot90(p, seed=rng.integers(1e9)),
                                           seed=rng.integers(1e9)),
                               seed=rng.integers(1e9))
            for p in patches
        ])
        out.append(aug)
    return np.concatenate(out, axis=0)
