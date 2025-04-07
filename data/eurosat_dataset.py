import torch
from torch.utils.data import Dataset
from glob import glob
from PIL import Image


class EuroSATDataset(Dataset):
    def __init__(self, root_dir, split, transforms):
        self.root_dir = root_dir
        self.split = split
        self.transforms = transforms

        categories = [x.split("/")[-1] for x in sorted(glob(f"{self.root_dir}/*"))]

        self.cat_to_label = {x: idx for idx, x in enumerate(categories)}
        self.label_to_cat = {idx: x for idx, x in enumerate(categories)}

        images_per_category = 3000
        train_images_count = int(0.8 * 3000)
        val_images_count = images_per_category - train_images_count

        self.images_per_category = {}
        for cat in categories:
            self.images_per_category[cat] = glob(f"{self.root_dir}/{cat}/*.jpg")

        self.images_arr = []
        if split == "train":
            for cat in categories:
                self.images_arr.extend(self.images_per_category[cat][:train_images_count])
        elif split == "val":
            for cat in categories:
                self.images_arr.extend(self.images_per_category[cat][-val_images_count:])

    def __len__(self):
        return len(self.images_arr)

    def __getitem__(self, index):
        img_path = self.images_arr[index]
        category = img_path.split("/")[-2]
        label = torch.tensor(self.cat_to_label[category], dtype=torch.int64)

        image = Image.open(img_path)

        if self.transforms is not None:
            image = self.transforms(image)

        return image, label


# if __name__ == "__main__":
#     t = transforms.ToTensor()
#     train_ds = EuroSATDataset(root_dir="./EuroSAT_RGB", split="train", transforms=t)
#     val_ds = EuroSATDataset(root_dir="./EuroSAT_RGB", split="val", transforms=t)

#     for x in range(len(val_ds)):
#         print(val_ds.__getitem__(x)[-1])

#     ipdb.set_trace()
