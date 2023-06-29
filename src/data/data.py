import torch
from torch.nn.functional import pad
from torch.utils.data import Dataset
from torchvision.io import read_image


class PulpExposureDataset(Dataset):
    def __init__(self, data, transform=lambda x: x):
        """
        :param data: np.array featuring one sample per row
        :param transform: transform function to be applied to both the input and the output
        """
        self.data = data
        self.transform = transform

    def __getitem__(self, i):
        # Process sample
        id, target, treatment_type, caries_pulp_distance, _ = self.data[i]

        # Load image
        image = read_image(f"data/final/{int(id)}/image_2.png").type(torch.float32) / 255

        # Augment image
        image = self.transform(image)
        image = image[0, :, :].unsqueeze(0).contiguous()

        # Prepare tensors
        target = torch.tensor([target], dtype=torch.float32)
        extra = torch.tensor([caries_pulp_distance, treatment_type], dtype=torch.float32)

        return dict(image=image, extra=extra, target=target)

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        # Unpack batch
        images = [b["image"] for b in batch]
        targets = [b["target"] for b in batch]

        # Determine the max height and width
        max_h = max([img.shape[1] for img in images])
        max_w = max([img.shape[2] for img in images])

        # Pad all images to match the max height and width
        images = [pad(img, (0, max_w - img.shape[2], 0, max_h - img.shape[1])) for img in images]

        # Convert to tensor
        images = torch.stack(images)
        targets = torch.stack(targets)

        return {"image": images, "target": targets}
