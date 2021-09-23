import os
import shutil

val_file_dir = '../assets/VOC2012/ImageSets/Main/'
orig_file_dir = '../assets/VOC2012/JPEGImages'
annotations_dir = '../assets/VOC2012/Annotations'
dest_dir = '../assets/Sampled_DS'

validation_files = [asset_file if asset_file.endswith('_val.txt') else None
                    for asset_file in os.listdir(val_file_dir)]

for validation_file in filter(lambda filename: filename is not None, validation_files):
    cls = validation_file.split('_')[0]
    dest_sub_dir = dest_dir + '/' + cls

    os.makedirs(dest_sub_dir, exist_ok=True)

    with open(val_file_dir + validation_file) as f:
        selected = 0

        for line in f:
            splits = line.strip().split(' ')
            if (len(splits) == 3) and splits[-1] == '1':
                img_name = '/' + splits[0] + '.jpg'
                annotation_name = '/' + splits[0] + '.xml'
                shutil.copy(orig_file_dir + img_name, dest_sub_dir + img_name)
                shutil.copy(annotations_dir + annotation_name, dest_sub_dir + annotation_name)
                selected += 1

                if selected == 100:
                    break
