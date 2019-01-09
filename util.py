import torch
import os


def save_state(ckpt_dir, step, model, optimizer):
    save_state_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    save_filepath = os.path.join(ckpt_dir, "model-%d.pth" % step)
    torch.save(save_state_dict, save_filepath)


def load_state(ckpt_dir, step, model, optimizer=None):
    save_filepath = os.path.join(ckpt_dir, "model-%d.pth" % step)
    if os.path.isfile(save_filepath):
        checkpoint = torch.load(save_filepath)
        model.load_state_dict(checkpoint["model"])
        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer"])
        print("=> restore checkpoint from %s finished!" % save_filepath)
    else:
        print("=> no checkpoint found at %s." % save_filepath)


def test_model(test_dataloader, label_map, model):
    with torch.no_grad():
        model.eval()
        num_correct, num_total = 0, 0
        for images, labels, label_lens in test_dataloader:
            images = images.cuda()
            outputs = model(images)
            preds = outputs.argmax(-1).cpu().numpy()

            pred_strs = label_map.decode(preds, raw=False)
            label_strs = label_map.decode_label(labels, label_lens)
            for pred_str, label_str in zip(pred_strs, label_strs):
                if pred_str == label_str:
                    num_correct += 1
                num_total += 1
        model.train()
    accuracy = num_correct / num_total
    print("=> test accuracy: %.4f" % accuracy)
    return accuracy

