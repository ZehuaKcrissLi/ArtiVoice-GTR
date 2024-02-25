import os

def change_names(in_file, out_file, prefix):
    with open(out_file, "w") as out:
        with open(in_file, "r") as f:
            for line in f:
                paths = line.strip().split("/")
                path = os.path.join(prefix, *paths[2:])
                out.write(path + "\n")


if __name__ == "__main__":
    # in_file = "Data/train_list_aishell3.txt"
    # out_file = "Data/train_list_aishell3_renamed.txt"
    in_file = "Data/val_list_aishell3.txt"
    out_file = "Data/val_list_aishell3_renamed.txt"
    prefix = "/storageNVME/melissa/aishell3"
    change_names(in_file, out_file, prefix)
