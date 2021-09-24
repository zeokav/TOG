import ast
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np

plt.clf()
fig = plt.figure(figsize=(15, 5))

ssd_mislabel_file = './10-iter/ssd_mislabel_out.txt'
yolo_mislabel_file = './10-iter/yolo_mislabel_out.txt'

starts = [
    'Now processing',
    'Benign Start',
    'Labels',
    'Benign End',
    'Attack Start',
    'Attack End'
]

divider = '---'
base_sample_dir = '../assets/Sampled_DS/'

ssd_entries = {}
yolo_entries = {}


def finalize(recording):
    file_name = recording['file']
    original_file_xml = base_sample_dir + file_name.replace('.jpg', '.xml')

    doc = ET.parse(original_file_xml)
    root = doc.getroot()
    recording['true_tags'] = []
    for object in root.findall('object'):
        recording['true_tags'].append(object.find('name').text)


def process(t, file):
    recording = {}
    for line in open(file):
        if divider in line:
            recording = {}
        if starts[0] in line:
            recording['file'] = line.split(':')[1].strip()
        if starts[1] in line:
            recording['ben_start'] = float(line.split(':')[1].strip())
        if starts[2] in line:
            if recording.get('ben_labels') is None:
                recording['ben_labels'] = ast.literal_eval(line.split(':')[1].strip())
            else:
                recording['att_labels'] = ast.literal_eval(line.split(':')[1].strip())
        if starts[3] in line:
            recording['ben_end'] = float(line.split(':')[1].strip())
        if starts[4] in line:
            recording['att_start'] = float(line.split(':')[1].strip())
        if starts[5] in line:
            recording['att_end'] = float(line.split(':')[1].strip())

            finalize(recording)
            if t == 'ssd':
                ssd_entries[recording['file']] = recording
            else:
                yolo_entries[recording['file']] = recording


def extract_divergence():
    counts_ssd = {}
    counts_yolo = {}
    for file in yolo_entries.keys():
        if 'dog' not in file or file not in ssd_entries:
            continue

        for label in yolo_entries.get(file).get('att_labels'):
            counts_yolo[label] = counts_yolo.get(label, 0) + 1

        for label in ssd_entries.get(file).get('att_labels'):
            counts_ssd[label] = counts_ssd.get(label, 0) + 1

    all_classes = sorted(set(counts_ssd.keys()).union(set(counts_yolo.keys())))
    ind = np.arange(len(all_classes))
    width = 0.35

    plt.xticks(ind + width / 2, all_classes)

    plt.title("Divergence of 'dog' across YOLO and SSD")
    plt.bar(ind, [counts_yolo.get(key, 0) for key in all_classes], width, label='YOLO')
    plt.bar(ind + width, [counts_ssd.get(key, 0) for key in all_classes], width, label='SSD')

    plt.show()


if __name__ == '__main__':
    process('ssd', ssd_mislabel_file)
    process('yolo', yolo_mislabel_file)

    extract_divergence()