import csv
import os
import os.path
import tarfile
from urllib.parse import urlparse

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import pickle
import util
from util import *
from tqdm import *

object_categories = ['bridge floor system',
'superstructure',
'substructure',
'bridge deck pavement',
'expansion joint',
'railing',
'drainage system',
'bridge plaque',
'beam',
'hinge joint',
'bearing',
'cap beam',
'pier',
'abutment',
'conical slope or slope protection',
'crack',
'spall',
'rebar',
'corrosion',
'speckle',
'repaired',
'blockage',
'information incomplete',
'joint mortar falling',
'shear deformation',
'separation',
'movement',
'weed',
'slope instability']


def read_image_label(file):
    print('[dataset] read ' + file)
    data = dict()
    with open(file, 'r') as f:
        for line in f:
            tmp = line.split(' ')
            name = tmp[0]
            label = int(tmp[-1])
            data[name] = label
            # data.append([name, label])
            # print('%s  %d' % (name, label))
    return data


def read_object_labels(root, set):
    path_labels = os.path.join(root, 'files')  
    labeled_data = dict()
    num_classes = len(object_categories)

    for i in range(num_classes):
        file = os.path.join(path_labels, object_categories[i] + '_' + set + '.txt')
        data = read_image_label(file)

        if i == 0:
            for (name, label) in data.items():
                labels = np.zeros(num_classes)
                labels[i] = label
                labeled_data[name] = labels
        else:
            for (name, label) in data.items():
                labeled_data[name][i] = label

    return labeled_data


def write_object_labels_csv(file, labeled_data):
    # write a csv file
    print('[dataset] write file %s' % file)
    with open(file, 'w') as csvfile:
        fieldnames = ['name']
        fieldnames.extend(object_categories)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for (name, labels) in labeled_data.items():
            example = {'name': name}
            for i in range(20):
                example[fieldnames[i + 1]] = int(labels[i])
            writer.writerow(example)

    csvfile.close()


def read_object_labels_csv(file, header=True):
    images = []
    num_categories = 0
    print('[dataset] read', file)
    with open(file, 'r') as f:
        reader = csv.reader(f)
        rownum = 0
        for row in reader:
            if header and rownum == 0:
                header = row
            else:
                if num_categories == 0:
                    num_categories = len(row) - 1
                name = row[0]
                labels = (np.asarray(row[1:num_categories + 1])).astype(np.float32)
                labels = torch.from_numpy(labels)
                item = (name, labels)
                images.append(item)
            rownum += 1
    return images


def find_images_classification(root, set):
    path_labels = os.path.join(root, 'files')  # or just 'root' if labels are there
    file = os.path.join(path_labels, set + '.txt')
    images = []
    with open(file, 'r') as f:
        for line in f:
            images.append(line.strip())  # remove trailing \n
    return images


class Classification(data.Dataset):
    def __init__(self, root, set, transform=None, target_transform=None, inp_name=None, adj=None):
        self.root = root
        self.path_images = os.path.join('dataset', 'JPEGImages')
        self.set = set
        self.transform = transform
        self.target_transform = target_transform
        self.imgs = []
        # download dataset

        # define path of csv file
        path_csv = os.path.join(self.root, 'files')
        # define filename of csv file
        file_csv = os.path.join(path_csv, 'classification_' + set + '.csv')

        # create the csv file if necessary
        if not os.path.exists(file_csv):
            if not os.path.exists(path_csv):  # create dir if necessary
                os.makedirs(path_csv)
            # generate csv file
            labeled_data = read_object_labels(self.root, self.set)
            # write csv file
            write_object_labels_csv(file_csv, labeled_data)

        self.classes = object_categories
        self.images = read_object_labels_csv(file_csv)
        #self.imgs = pickle.load(open(str(set)+'.pkl', 'rb'))
        for imgpath in tqdm(self.images):
            img = Image.open(os.path.join(self.path_images, imgpath[0] + '.jpg')).convert('RGB').resize((448, 448))
            self.imgs.append(img)
            pass
        #pickle.dump(self.imgs, open(str(set)+'.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)

        pass


        with open(inp_name, 'rb') as f:
            self.inp = pickle.load(f)
        self.inp_name = inp_name

        print('[dataset] classification set=%s number of classes=%d  number of images=%d' % (
            set, len(self.classes), len(self.images)))

    def __getitem__(self, index):
        path, target = self.images[index]
        pool_name = 'dataset/pool_pkls/'+ path + '.pkl'
        with open(pool_name, 'rb') as f:
            pool = pickle.load(f)
        #img = Image.open(os.path.join(self.path_images, path + '.jpg')).convert('RGB')
        img = self.imgs[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return (img, path, pool), target

    def __len__(self):
        return len(self.images)

    def get_number_classes(self):
        return len(self.classes)
