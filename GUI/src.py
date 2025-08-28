import numpy as np
import torch
from torch import nn

class BinaryClassifier(nn.Module):
    def __init__(self, input_size=2048):
        super(BinaryClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output probability between 0 and 1
        )

    def forward(self, x):
        return self.fc(x)

def load_model(model_path, input_size=2048):
    model = BinaryClassifier(input_size)

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()  # Set model to evaluation mode
    return model

def randomly_place_cells(out_size, rgb_images, mask_images, targets, n_images, max_rejections=10):
    
    hpsz = rgb_images.shape[1] // 2  
    out_image = np.zeros([*np.array(out_size) + 2*hpsz, 3])
    mask_image = np.zeros([*np.array(out_size) + 2*hpsz])
    
    rejection_count = 0  # Counter for consecutive rejections
    placed_cells = 0
    placed_idxs, placed_pos, placed_targets = [], [], []
    # Loop until all cells are placed or too many rejections occur
    while placed_cells < n_images:
        # Randomly select a cell and calculate its diameter
        if rejection_count == max_rejections:
            print(f'reached max rejections after {placed_cells} cells')
            break
        
        cell_index = np.random.choice(len(rgb_images))

        # Generate a random position
        try_pos = [np.random.randint(hpsz, out_size[0] + hpsz), np.random.randint(hpsz, out_size[1] + hpsz)]
        
        if mask_image[try_pos[0], try_pos[1]] == 1:
            rejection_count += 1 
            continue

        # Check if the new position overlaps with any existing cell
        tmp = np.zeros_like(mask_image)
        rot = np.random.randint(3)
        tmp[try_pos[0]-hpsz:try_pos[0]+hpsz, try_pos[1]-hpsz:try_pos[1]+hpsz] = np.rot90(mask_images[cell_index], k=rot)
        if np.any((mask_image + tmp) > 1):
            rejection_count += 1 
            continue
        
        mask_image = mask_image + tmp
        
        out_image[
            try_pos[0]-hpsz:try_pos[0]+hpsz,
            try_pos[1]-hpsz:try_pos[1]+hpsz
        ] += np.rot90(rgb_images[cell_index], k=rot)

        placed_cells += 1
        
        placed_idxs.append(cell_index)
        placed_pos.append([try_pos[0]-hpsz, try_pos[1]-hpsz])
        placed_targets.append(targets[cell_index])

    return np.clip(out_image[hpsz:-hpsz, hpsz:-hpsz], 0, 1), mask_image[hpsz:-hpsz, hpsz:-hpsz], placed_idxs, placed_pos, placed_targets