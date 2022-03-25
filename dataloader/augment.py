import albumentations as A
from albumentations.pytorch import ToTensorV2


class Transform:
    def __init__(self, mode=None):
        assert mode in ("train", "val", "debug")
        if mode == "train":
            self.trans = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2()
            ])
        elif mode == "val":
            self.trans = A.Compose([
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2()
            ])
        else:
            self.trans = A.Compose([
                A.HorizontalFlip(p=0.5),
                # A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2()
            ])

    def __call__(self, image):
        data = self.trans(image=image)["image"]
        return data
