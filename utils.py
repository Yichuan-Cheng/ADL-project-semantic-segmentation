import os
from PIL import Image

def get_files(folder, name_filter=None, extension_filter=None):
    if not os.path.isdir(folder):
        raise RuntimeError("\"{0}\" is not a folder.".format(folder))
    if name_filter is None:
        name_cond = lambda filename: True
    else:
        name_cond = lambda filename: name_filter in filename
    if extension_filter is None:
        ext_cond = lambda filename: True
    else:
        ext_cond = lambda filename: filename.endswith(extension_filter)
    filtered_files = []
    for path, _, files in os.walk(folder):
        files.sort()
        for file in files:
            if name_cond(file) and ext_cond(file):
                full_path = os.path.join(path, file)
                filtered_files.append(full_path)
    return filtered_files

def pil_loader(data_path, label_path):
    data = Image.open(data_path)
    label = Image.open(label_path)
    return data, label
