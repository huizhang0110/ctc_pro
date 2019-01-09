import torch
from torch.utils import data
from torch.utils.data import sampler
import os
import pickle
from tqdm import tqdm
from PIL import Image, ImageFile
import numpy as np
from torchvision import transforms
import random
ImageFile.LOAD_TRUNCATED_IMAGES = True


class LabelMap(object):
    def __init__(self, character_set=None):
        if character_set is None:
            character_set = list("0123456789abcdefghijklmnopqrstuvwxyz")
        self.character_set = character_set
        self.char_to_label_map = {}
        self.char_to_label_map["-"] = 0 # 0 is reserved for blank required by warp_ctc
        for i, c in enumerate(self.character_set):
            self.char_to_label_map[c] = i + 1
        self.label_to_char_map = {v:k for k, v in self.char_to_label_map.items()}

    @property
    def num_classes(self):
        return len(self.character_set) + 1

    def encode(self, text):
        if isinstance(text, str):
            label = [self.char_to_label_map[c] for c in text.strip().lower()]
        else:
            raise TypeError("Not support this type!")
        return label

    def decode(self, ids, raw=True):
        if not isinstance(ids, np.ndarray):
            raise TypeError("indices must be np.ndarray type, but got %s" % type(ids))
        if ids.ndim == 1:
            if raw:
                chars = [self.label_to_char_map[i] for i in ids]
            else:
                chars = []
                for i, x in enumerate(ids):
                    if x == 0 or (i > 0 and x == ids[i - 1]):
                        continue
                    chars.append(self.label_to_char_map[x])
            return "".join(chars)
        else:
            words = [self.decode(i, raw=raw) for i in ids]
            return words
        
    def decode_label(self, ids, lens):
        if not (isinstance(ids, torch.Tensor) and isinstance(lens, torch.Tensor)):
            raise TypeError("ids and lens must be torch.Tensor type, but got %s" % type(ids))
        if lens.numel() == 1:
            length = lens[0]
            assert ids.numel() == length, "text with length: %d " \
                    "does not match declared length %d" % (ids.numel(), length)
            chars = [self.label_to_char_map[i.item()] for i in ids]
            return "".join(chars)
        else: # batch
            assert ids.numel() == lens.sum(), "texts with length %d " \
                    "does not match declared length %d" % (ids.numel(), lens.sum())
            texts = []
            index = 0
            for i in range(lens.numel()):
                l = lens[i]
                texts.append(self.decode_label(ids[index:index+l], torch.IntTensor([l])))
                index += l
            return texts


class TagsDataset(data.Dataset):

    def __init__(self, tags_file, label_map, transform):
        super(TagsDataset, self).__init__()
        self.tags_file = tags_file
        self.label_map = label_map
        self.cache_file = ".cache_%s.pkl" % (os.path.basename(tags_file).split(".")[0])
        self.transform = transform
        self.image_names = []
        self.labels = []
        self.data_root = os.path.dirname(tags_file)
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                self.image_names, self.labels = pickle.load(f)
        else:
            with open(self.tags_file, "r") as f:
                lines = f.read().strip().split("\n")
                for line in tqdm(lines):
                    image_name, gt = line.split(" ", 1)
                    label = self.label_map.encode(gt)
                    if not os.path.exists(os.path.join(self.data_root, image_name)):
                        print("Skip %s" % line)
                        continue
                    self.image_names.append(image_name)
                    self.labels.append(label)
            with open(self.cache_file, "wb") as f:
                pickle.dump((self.image_names, self.labels), f)
            print("Initial dataset from %s finished!" % self.tags_file)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        try:
            image_path = os.path.join(self.data_root, self.image_names[idx])
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print("Error image: ", image_path)
            return self[(idx + 1) % len(self)]

        if self.transform:
            image = self.transform(image)
        target = np.array(self.labels[idx])
        return image, torch.IntTensor(target)


def collate_fn(image_label_pair):
    images, labels = zip(*image_label_pair)
    b_images = torch.stack(images)
    lens = [len(label) for label in labels]
    b_lens = torch.IntTensor(lens)
    b_labels = torch.cat(labels)
    return b_images, b_labels, b_lens


if __name__ == "__main__":
    label_map = LabelMap()
    transform = transforms.Compose([
        transforms.Resize([32, 100]),
        transforms.ToTensor()])
    dataset = TagsDataset(
            tags_file="./benchmarks/IIIT5k/IIII5k.tags",
            label_map=label_map,
            transform=transform)
    dataloader = data.DataLoader(
            dataset=dataset,
            batch_size=10,
            shuffle=True,
            num_workers=1,
            collate_fn=collate_fn)

    for images, labels, label_lens in dataloader:
        print(images.shape)
        print(labels)
        print(label_lens)
        strs = label_map.decode_label(labels, label_lens)
        print(strs)
        exit()
