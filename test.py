import torch
from util import load_state, test_model
from input_data import LabelMap, TagsDataset, collate_fn
from torchvision import transforms
from torch.utils.data import DataLoader
from model import CTCModel


# test_file = "./benchmarks/IIIT5k/IIII5k.tags"
# test_file = "./benchmarks/ic13/ic13_1015.tags"
test_file = "./benchmarks/svt/svt.tags"
ckpt_dir = "./experiments/master"
step = 330000


label_map = LabelMap()
n_classes = label_map.num_classes
test_transform = transforms.Compose([
    transforms.Resize((32, 128)),
    transforms.ToTensor()])
test_dataset = TagsDataset(test_file, label_map, test_transform)
test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn)


network = CTCModel(n_classes)
network = network.cuda()
load_state(ckpt_dir, step, network)

test_model(test_dataloader, label_map, network, print_result=True)

