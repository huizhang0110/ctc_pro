import torch
from warpctc_pytorch import CTCLoss
import torch.optim as optim
from model import CTCModel
from input_data import LabelMap, TagsDataset, collate_fn
from torch.utils.data import DataLoader
from util import save_state, load_state, test_model
from torchvision import transforms


seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# === Configure ====
init_learning_rate = 1.0
weight_decay = 1e-4
n_hidden = 512
print_every = 100
sample_every = 500
save_state_every = 10000
debug = False
if debug:
    tags_file = "/home/zhui/icdar/data/train_data/debug_data/gt.tags"
    batch_size = 10
    num_workers = 1
    ckpt_dir = "experiments/debug"
else:
    tags_file = "/share/zhui/mnt/ramdisk/max/90kDICT32px/synth90k_all_sorted.tags"
    test_file = "benchmarks/IIIT5k/IIII5k.tags"
    batch_size = 64
    num_workers = 16
    ckpt_dir = "experiments/master"

# ==== data input pipeline ====
label_map = LabelMap()
n_classes = label_map.num_classes
transform = transforms.Compose([
    transforms.Resize((32, 128)),
    transforms.ToTensor()])
train_dataset = TagsDataset(tags_file, label_map, transform)
train_dataloader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers,
        collate_fn=collate_fn)

test_dataset = TagsDataset(test_file, label_map, transform)
test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn)

# ==== Building network ====
model = CTCModel(n_classes)
model = model.cuda()
ctc_loss = CTCLoss(size_average=True).cuda()
optimizer = optim.Adadelta(model.parameters(), lr=init_learning_rate)

# ==== Training progress ====
print("=> Begining training!")
model.train()
step = 1
if step != 1:
    load_state(ckpt_dir, step, model, optimizer)
total_loss = 0

while 1:
    for images, labels, label_lens in train_dataloader:
        images = images.cuda()

        if step % sample_every == 0:
            with torch.no_grad():
                model.eval()
                print("\n", "--" * 20)
                output = model(images) # b, t, a
                print("output: ")
                pred = output.argmax(-1).cpu().numpy() # b,t
                print(label_map.decode(pred, raw=False))
                print("label: ")
                print(label_map.decode_label(labels, label_lens))
                print("--" * 20)
                model.train()

        optimizer.zero_grad()
        output = model(images)
        probs = output.transpose(0, 1).contiguous().cuda()
        label_size = label_lens
        probs_size = torch.IntTensor([probs.size(0)] * probs.size(1))
        probs.requires_grad_(True)
        loss = ctc_loss(probs, labels, probs_size, label_size)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if step % print_every == 0:
            print("step: %d, loss: %.5f" % (step, total_loss / print_every))
            total_loss = 0

        if step % save_state_every == 0:
            save_state(ckpt_dir, step, model, optimizer)
            accuracy = test_model(test_dataloader, label_map, model)
            
        step += 1

