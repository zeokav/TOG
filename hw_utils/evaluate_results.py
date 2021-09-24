import ast
import sys

import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np

# plt.style.use('ggplot')
# plt.switch_backend()
plt.clf()
fig = plt.figure(figsize=(8, 10))

ind = np.arange(3)
width = 0.35
cols = ['iter-1', 'iter-3', 'iter-10']

benign_recalls = []
attack_recalls = []

benign_times = []
attack_times = []

plt.xticks(ind + width/2, cols)


roots = {
    1: './1-iter',
    3: './3-iter',
    10: './10-iter'
}

out_files = {
    'SSD': {
        "ML": "ssd_mislabel_out.txt",
        "VA": "ssd_vanish_out.txt"
    },
    'YOLO': {
        "ML": "yolo_mislabel_out.txt",
        "VA": "yolo_vanish_out.txt"
    }
}


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


def finalize(recording):
    file_name = recording['file']
    original_file_xml = base_sample_dir + file_name.replace('.jpg', '.xml')

    doc = ET.parse(original_file_xml)
    root = doc.getroot()
    recording['true_tags'] = []
    for object in root.findall('object'):
        recording['true_tags'].append(object.find('name').text)


def plot_accuracy(entries):
    benign_correct = 0
    attack_correct = 0

    benign_time_taken = 0
    attack_time_taken = 0

    tot = len(entries)
    for entry in entries:
        true_tags = entry['true_tags']
        true_tags.sort()

        benign_tags = entry['ben_labels']
        benign_tags.sort()

        att_tags = entry['att_labels']
        att_tags.sort()

        benign_time_taken = entry['ben_end'] - entry['ben_start']
        attack_time_taken = entry['att_end'] - entry['att_start']

        if true_tags == benign_tags:
            benign_correct += 1

        if true_tags == att_tags:
            attack_correct += 1

    benign_recalls.append(benign_correct/tot)
    attack_recalls.append(attack_correct/tot)

    benign_times.append(benign_time_taken/tot)
    attack_times.append(attack_time_taken/tot)


def run_for_iterations(n_iter):
    log_file_path = '{}/{}'.format(roots[n_iter], out_files[model][attack])

    all_entries = []

    recording = {}
    for line in open(log_file_path):
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
            all_entries.append(recording)

    plot_accuracy(all_entries)

if __name__ == '__main__':
    model = sys.argv[1]
    attack = sys.argv[2]

    run_for_iterations(1)
    run_for_iterations(3)
    run_for_iterations(10)

    print("MODEL: {} | ATTACK: {}".format(model, attack))
    for iteration in range(3):
        print('----- {} -----'.format(cols[iteration]))
        print('Benign Recall (RB): {}\nAttacked Recall (RA): {}\nEffectiveness % ((RB-RA)/RB): {}\n'.format(benign_recalls[iteration], attack_recalls[iteration], (benign_recalls[iteration] - attack_recalls[iteration]) * 100/benign_recalls[iteration]))
        print('Benign Run Time (RB): {}\nAttack Run Time (RA): {}\nIncrease Factor(RA/RB): {}\n'.format(benign_times[iteration], attack_times[iteration], attack_times[iteration]/benign_times[iteration]))

    plt.bar(ind, benign_recalls, width, label='Benign Recall')
    plt.bar(ind+width, attack_recalls, width, label='Attacked Recall')
    plt.plot(ind, [100 * t for t in benign_times], label='Benign Run Time (Scaled)')
    plt.plot(ind+width, [100 * t for t in attack_times], label='Attack Run Time (Scaled)')

    plt.legend()
    plt.show()
