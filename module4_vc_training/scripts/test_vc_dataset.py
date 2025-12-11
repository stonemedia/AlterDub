import os
from torch.utils.data import DataLoader

from module4_vc_training.datasets.vc_dataset import VCDataset, vc_collate_fn


def find_train_list() -> str:
    """
    Try a couple of common locations for train_list.txt relative to project root.
    Adjust this if your file lives somewhere else.
    """
    # Project root = two levels up from this file: D:\AlterDub
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))

    candidates = [
        os.path.join(root_dir, "module3_dataset", "train_list.txt"),
        os.path.join(root_dir, "module3_dataset", "metadata", "train_list.txt"),
    ]

    for path in candidates:
        if os.path.isfile(path):
            print(f"[INFO] Using train_list at: {path}")
            return path

    # If nothing found, print all candidates and raise
    msg_lines = ["[ERROR] Could not find train_list.txt. Tried:"]
    for path in candidates:
        msg_lines.append(f"  - {path}")
    raise FileNotFoundError("\n".join(msg_lines))


def main():
    list_path = find_train_list()
    segment_frames = 128

    ds = VCDataset(list_path=list_path, segment_frames=segment_frames, is_train=True)
    print(f"Dataset size: {len(ds)}")

    loader = DataLoader(
        ds,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=vc_collate_fn,
    )

    batch = next(iter(loader))
    print("mel:", batch["mel"].shape)        # (B, T_max, n_mels)
    print("f0:", batch["f0"].shape)          # (B, T_max)
    print("spk_id:", batch["spk_id"].shape)  # (B,)
    print("lengths:", batch["lengths"])
    print("mask:", batch["mask"].shape)


if __name__ == "__main__":
    main()
