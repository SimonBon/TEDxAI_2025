
from scipy.ndimage import gaussian_filter, map_coordinates
import torchvision.transforms as T
from copy import deepcopy
import numpy as np
import torch
import cv2

from mmselfsup.registry import TRANSFORMS
from mmcls.registry import TRANSFORMS as CLS_TRANSFORMS
from mmcv.transforms.base import BaseTransform

@CLS_TRANSFORMS.register_module()
@TRANSFORMS.register_module()
class C_RandomAffine(BaseTransform):
    
    def __init__(self, angle=(0,360), scale=(0.9, 1.1), shift=(-0.1,0.1), order=0):
        super().__init__()
        
        self.angle = angle
        self.scale = scale
        self.shift = shift
        self.order = order

        assert self.angle[0] <= self.angle[1], f'angle[0]: {angle[0]} must be smaller or equal to angle[1]: {angle[1]}'
        assert self.scale[0] <= self.scale[1], f'scale[0]: {scale[0]} must be smaller or equal to scale[1]: {scale[1]}'
        assert self.shift[0] <= self.shift[1], f'shift[0]: {shift[0]} must be smaller or equal to shift[1]: {shift[1]}'

    #@Timer(name='C_RandomAffine', text='Function '{name}' took {seconds:.6f} seconds to execute.')        
    def transform(self, results: dict) -> dict:
        '''Randomly crop the image and resize the image to the target size.

        Args:
            results (dict): Result dict from previous pipeline.

        Returns:
            dict: Result dict with the transformed image.
        '''
        
        img = results['img']
        height, width = img.shape[:2]

        # Random scaling
        curr_scale = np.random.uniform(self.scale[0], self.scale[1])

        # Random translation
        shift_y = np.random.uniform(self.shift[0], self.shift[1])
        shift_x = np.random.uniform(self.shift[0], self.shift[1])

        # Random rotation
        curr_angle = np.random.uniform(self.angle[0], self.angle[1])

        # Compute the combined transformation matrix
        center = (width / 2, height / 2)

        # Scaling matrix
        scale_matrix = cv2.getRotationMatrix2D(center, 0, curr_scale)

        # Translation matrix
        translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])

        # Rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, curr_angle, 1)

        # Combine the transformation matrices
        transform_matrix = scale_matrix
        transform_matrix[0, 2] += translation_matrix[0, 2]
        transform_matrix[1, 2] += translation_matrix[1, 2]

        # Apply the combined transformation matrix
        img = cv2.warpAffine(img, transform_matrix, (width, height), flags=self.order, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # Apply the rotation after scaling and translation
        img = cv2.warpAffine(img, rotation_matrix, (width, height), flags=self.order, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        results['img'] = img
        results['angle'] = curr_angle
        results['scale'] = curr_scale
        results['shift'] = (shift_x, shift_y)
        
        return results


@CLS_TRANSFORMS.register_module()
@TRANSFORMS.register_module()   
class C_RandomNoise(BaseTransform):
    
    def __init__(self, mean=(0, 0), std=(0, 0.07), clip=True):
        super().__init__()
        
        self.mean = mean
        assert self.mean[0] <= self.mean[1]
        self.std = std
        assert self.std[0] <= self.std[1]
        
        self.clip = clip

    #@Timer(name='RandomNoise', text='Function '{name}' took {seconds:.6f} seconds to execute.')    
    def transform(self, results: dict) -> dict:
        '''Add random noise to the image with a mean and std deviation chosen randomly within the specified range.

        Args:
            results (dict): Result dict from the previous pipeline.

        Returns:
            dict: Result dict with the transformed image.
        '''
        
        img = results['img']
        
        # Randomly choose mean and std within the given range
        curr_mean = np.random.uniform(self.mean[0], self.mean[1])
        curr_std = np.random.uniform(self.std[0], self.std[1])
        
        # Add Gaussian noise to the image
        noise = np.random.normal(curr_mean, curr_std, img.shape)
        img = img + noise
        
        # Clip the values to be in the valid range
        if self.clip:
            if img.dtype == np.uint8:
                img = np.clip(img, 0, 255)
            else:  # assuming float type
                img = np.clip(img, 0.0, 1.0)
        
        results['img'] = img
        results['noise_level'] = (curr_mean, curr_std)
        
        return results


@CLS_TRANSFORMS.register_module()
@TRANSFORMS.register_module()
class C_RandomGradient(BaseTransform):
    def __init__(self, low=(0, 0), high=(1, 1), clip: bool = True, threshold=0.0):
        super().__init__()
        self.low = low
        self.high = high
        self.clip = clip
        self.threshold = threshold

    def transform(self, results: dict) -> dict:
        """
        Randomly apply a gradient to the non-zero regions of the image.
        Keeps zero-valued pixels unchanged.
        """
        img = results['img']
        img = img.astype(np.float32, copy=False)  # ensure float

        # Determine channel count
        if img.ndim == 2:
            img = img[..., None]  # make it HWC with 1 channel
        n_channels = img.shape[-1]

        # Create a non-zero mask (True where pixel != 0 in any channel)
        mask = np.any(img >= self.threshold, axis=-1, keepdims=True).astype(np.float32)

        gradient_images = []
        for _ in range(n_channels):
            curr_low = np.random.uniform(self.low[0], self.low[1])
            curr_high = np.random.uniform(self.high[0], self.high[1])

            theta = np.random.rand() * 2 * np.pi
            direction = np.array([np.cos(theta), np.sin(theta)])

            XX, YY = np.meshgrid(
                np.linspace(-1, 1, img.shape[0]),
                np.linspace(-1, 1, img.shape[1]),
            )
            directed_image = XX * direction[0] + YY * direction[1]
            directed_image = (directed_image - directed_image.min()) / (directed_image.max() - directed_image.min())
            scaled_directed_image = (curr_high - curr_low) * directed_image + curr_low

            gradient_images.append(scaled_directed_image)

        gradient_images = np.stack(gradient_images, axis=-1)

        # Apply gradient only where mask == 1
        out_img = img + gradient_images * mask

        results['img'] = np.clip(out_img, 0, 1) if self.clip else out_img
        return results


@CLS_TRANSFORMS.register_module()
@TRANSFORMS.register_module()
class C_RandomFlip(BaseTransform):
    
    def __init__(self, p_vertical: float = 0.5, p_horizontal: float = 0.5):
        super().__init__()
        
        self.p_vertical = p_vertical
        self.p_horizontal = p_horizontal
    
    def transform(self, results: dict) -> dict:
        img = results['img']

        # vertical flip
        if np.random.random() > self.p_vertical:
            img = img[::-1, :, :]
        # horizontal flip
        if np.random.random() > self.p_horizontal:
            #if random.choice([True, False]):
            img = img[:, ::-1, :]

        results['img'] = img
        return results


@CLS_TRANSFORMS.register_module()
@TRANSFORMS.register_module()
class C_RandomBlurr(BaseTransform):
    
    def __init__(self, blurr: tuple = (0, 1), clip: bool = True):
        super().__init__()
        self.blurr = blurr
        self.clip = clip

    def transform(self, results: dict) -> dict:
        img = results['img']
        sigma = np.random.uniform(self.blurr[0], self.blurr[1]) + 1e-6
        kernel_size = (31, 31)
        img = cv2.GaussianBlur(img, kernel_size, sigmaX=sigma, sigmaY=sigma)
        results['img'] = np.clip(img, 0, 1) if self.clip else img
        return results


@CLS_TRANSFORMS.register_module()
@TRANSFORMS.register_module()
class C_RandomIntensity(BaseTransform):
    
    def __init__(self, low=(0.5, 0.5, 0.5), high=(2, 2, 2)):
        super().__init__()
        
        self.low = low
        self.high = high
                
    def transform(self, results: dict) -> dict:
        img = results['img'].astype(np.float32)
        H, W, C = img.shape
        scaling_factors = np.random.uniform(self.low, self.high)

        # Apply scaling to each channel except the last one
        img[:, :, :-1] *= scaling_factors[:-1]

        results['img'] = np.clip(img, 0, 1)
        return results


@CLS_TRANSFORMS.register_module()
@TRANSFORMS.register_module()
class C_CentralCutter(BaseTransform):
    
    def __init__(self, size: int):
        super().__init__()
        
        assert (size%2) == 0
        self.hsz = size // 2

    def transform(self, results: dict) -> dict:
        img = results['img']
        
        c = img.shape[0] // 2, img.shape[1]//2
        
        #cut out the central part
        cropped_img = img[c[0]-self.hsz:c[0]+self.hsz, c[1]-self.hsz:c[1]+self.hsz]
        
        results['img'] = cropped_img
        results['cut_size'] = 2*self.hsz
        
        return results


@CLS_TRANSFORMS.register_module()
@TRANSFORMS.register_module()
class C_ElasticTransform(BaseTransform):
    
    def __init__(self, alpha: int = 700, sigma: int = 22):
        super().__init__()

        self.alpha = alpha
        self.sigma = sigma
    
    def transform(self, results) -> dict:
        img = results['img']
        W, H, C = img.shape

        dx = gaussian_filter(np.random.uniform(-1, 1, size=(W, H)), self.sigma, mode="constant") * self.alpha
        dy = gaussian_filter(np.random.uniform(-1, 1, size=(W, H)), self.sigma, mode="constant") * self.alpha

        x, y = np.meshgrid(np.arange(W), np.arange(H))
        indices = (y + dy).flatten(), (x + dx).flatten()

        distorted_image = np.zeros_like(img)

        # Apply the same transformation to each channel
        for c in range(C):
            distorted_channel = map_coordinates(img[..., c], indices, order=0, mode='reflect').reshape((W, H))
            distorted_image[..., c] = distorted_channel

        distorted_image = gaussian_filter(distorted_image, sigma=(1, 1, 0))
        #distorted_image = (distorted_image / distorted_image.max())

        _, labels_im, stats, _ = cv2.connectedComponentsWithStats(deepcopy((img[..., 2]*255).astype(np.uint8)))
        try:
            largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # Skip the background (index 0)

            # Get centroid of the largest component
            centroid_x = stats[largest_component, cv2.CC_STAT_TOP] + stats[largest_component, cv2.CC_STAT_HEIGHT] // 2
            centroid_y = stats[largest_component, cv2.CC_STAT_LEFT] + stats[largest_component, cv2.CC_STAT_WIDTH] // 2

            # Calculate the shift needed to move the centroid to the center of the patch
            shift_x = W // 2 - centroid_x
            shift_y = H // 2 - centroid_y
        except:
            shift_x = 0
            shift_y = 0

        # Shift the images
        M = np.float32([[1, 0, shift_y], [0, 1, shift_x]])
        distorted_image = cv2.warpAffine(distorted_image, M, (H, W), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        results['img'] = distorted_image
        return results


@CLS_TRANSFORMS.register_module()
@TRANSFORMS.register_module()
class C_BlueRemover(BaseTransform):
    
    def __init__(self):
        super().__init__()

    def transform(self, results: dict) -> dict:
        img = results['img']
        h, w, _ = img.shape
        img[..., 2] = np.zeros((h, w), dtype=img.dtype)
        results['img'] = img
        return results


@CLS_TRANSFORMS.register_module()
@TRANSFORMS.register_module()
class C_ChannelRemover(BaseTransform):
    
    def __init__(self, channels: list):
        super().__init__()
        self.channels = channels

    def transform(self, results: dict) -> dict:
        img = results['img']
        h, w, _ = img.shape
        img[..., self.channels] = np.zeros((h, w, len(self.channels)), dtype=img.dtype)
        results['img'] = img
        return results


@CLS_TRANSFORMS.register_module()
@TRANSFORMS.register_module()
class C_TensorCombiner(BaseTransform):
    
    def __init__(self):
        super().__init__()

    def transform(self, results) -> dict:
        """Concatenate image and mask tensors along the last dimension.

        Args:
            results (dict): Result dictionary containing 'img' and 'masks'.

        Returns:
            dict: Updated result dictionary with concatenated image and mask tensor.
        """
        if results['masks']:
            img = results['img']
            masks = np.atleast_3d(np.array(results['masks'])).transpose(1,2,0)  
            concat_tensor = np.concatenate((img, masks), axis=-1)
            results['img'] = concat_tensor
        
        return results

_DTYPE_MAP_NUMPY = {
    "float32": np.float32,
    "float64": np.float64,
    "float16": np.float16,
    "uint8":   np.uint8,
    "int64":   np.int64,
    "int32":   np.int32,
}

_DTYPE_MAP_TORCH = {
    "float32": torch.float32,
    "float64": torch.float64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "uint8":   torch.uint8,
    "int64":   torch.int64,
    "int32":   torch.int32,
}

@CLS_TRANSFORMS.register_module()
@TRANSFORMS.register_module()
class C_TypeCaster(BaseTransform):
    """
    Cast results['img'] (and optionally extra keys) to a specific dtype.

    Args:
        dtype (str): target dtype ('float32', 'float16', 'float64', 'uint8', etc.)
        backend (str): 'numpy' (default) casts ndarray dtype; 'torch' casts tensor dtype.
        keys (list[str]): which keys to cast; default ['img'].
        normalize_uint8_to_01 (bool): if input is uint8 and target is float*, divide by 255.0.
    """
    def __init__(
        self,
        dtype: str = "float32",
        backend: str = "numpy",
        keys=None,
        normalize_uint8_to_01: bool = True,
    ):
        super().__init__()
        self.dtype = dtype
        self.backend = backend.lower()
        self.keys = keys if keys is not None else ["img"]
        self.normalize_uint8_to_01 = normalize_uint8_to_01

        if self.backend not in ("numpy", "torch"):
            raise ValueError("backend must be 'numpy' or 'torch'")

        if self.backend == "numpy" and self.dtype not in _DTYPE_MAP_NUMPY:
            raise ValueError(f"Unsupported numpy dtype '{self.dtype}'. Supported: {list(_DTYPE_MAP_NUMPY.keys())}")
        if self.backend == "torch" and self.dtype not in _DTYPE_MAP_TORCH:
            raise ValueError(f"Unsupported torch dtype '{self.dtype}'. Supported: {list(_DTYPE_MAP_TORCH.keys())}")

    def _cast_numpy(self, arr):
        if not isinstance(arr, np.ndarray):
            # leave non-arrays alone
            return arr
        tgt = _DTYPE_MAP_NUMPY[self.dtype]
        if self.normalize_uint8_to_01 and arr.dtype == np.uint8 and np.issubdtype(tgt, np.floating):
            arr = arr.astype(np.float32) / 255.0
            if tgt is not np.float32:
                arr = arr.astype(tgt)
            return arr
        return arr.astype(tgt, copy=False)

    def _cast_torch(self, x):
        if not torch.is_tensor(x):
            # convert to tensor first
            x = torch.as_tensor(x)
        tgt = _DTYPE_MAP_TORCH[self.dtype]
        if self.normalize_uint8_to_01 and x.dtype == torch.uint8 and tgt.is_floating_point:
            x = x.float().div_(255.0).to(tgt)
            return x
        return x.to(dtype=tgt)

    def _cast_any(self, obj):
        if self.backend == "numpy":
            return self._cast_numpy(obj)
        return self._cast_torch(obj)

    def transform(self, results: dict) -> dict:
        for k in self.keys:
            if k not in results:
                continue
            val = results[k]
            if isinstance(val, list):
                results[k] = [self._cast_any(v) for v in val]
            else:
                results[k] = self._cast_any(val)
        return results