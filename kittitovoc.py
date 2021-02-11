###############################################################################
##########                        KITTI format                       ##########
"""
Values    Name      Description
----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.
"""
###############################################################################

# Import necessary libraries
import os, sys, shutil, glob, argparse
import numpy as np
from PIL import Image
from lxml import etree

python_version = sys.version_info.major


###########################################################
##########        KITTI to YOLO Conversion       ##########
###########################################################
def determine_label_yolo(label, labels):
    """
    Definition: Converts label to index in label set

    Parameters: label - label from file
                labels - list of labels
    Returns: index of label in labels (in str format)
    """
    return str(labels.index(label))


def parse_labels_yolo(label_file, labels, img_width, img_height):
    """
    Definition: Parses label files to extract label and bounding box
        coordinates.  Converts (x1, y1, x1, y2) KITTI format to
        (x, y, width, height) normalized YOLO format.

    Parameters: label_file - file with KITTI label(s) inside
                labels - list of labels in dataset
                img_width - width of input image
                img_height - height of input image
    Return: all_labels - contains a list of labels for objects in image
            all_coords - contains a list of coordinate for objects in image
    """
    lfile = open(label_file)
    coords = []
    all_coords = []
    all_labels = []
    for line in lfile:
        l = line.split(" ")
        all_labels.append(determine_label_yolo(l[0], labels))
        coords = list(map(int, list(map(float, l[4:8]))))
        x = float((float(coords[2]) + float(coords[0])) / 2.0) / float(img_width)
        y = float((float(coords[3]) + float(coords[1])) / 2.0) / float(img_height)
        width = float(float(coords[2]) - float(coords[0])) / float(img_width)
        height = float(float(coords[3]) - float(coords[1])) / float(img_height)
        tmp = [x, y, width, height]
        all_coords.append(tmp)
    lfile.close()
    return all_labels, all_coords


def copy_images_yolo(kitti, yolo):
    """
    Definition: Copy all images from the training and validation image sets
        in kitti format to training and validation image sets in yolo format.
        This means converting from .png to .jpg

    Parameters: kitti - path to kitti directory (contains 'Dataset\image_2' and 'Dataset\image_2_test')
                yolo - path to yolo output directory
    Returns: None
    """
    for filename in glob.glob(os.path.join(kitti + "Dataset/image_2/", "*.*")):
        shutil.copy(filename, yolo + "Dataset/image_2/")
    for filename in glob.glob(os.path.join(kitti + "Dataset/image_2_test/", "*.*")):
        shutil.copy(filename, yolo + "Dataset/image_2_test/")

    for filename in glob.glob(os.path.join(yolo + "Dataset/image_2/", "*.*")):
        im = Image.open(filename)
        im.save(filename.split(".png")[0] + ".jpg", "jpeg")
        os.remove(filename)
    for filename in glob.glob(os.path.join(yolo + "Dataset/image_2_test/", "*.*")):
        im = Image.open(filename)
        im.save(filename.split(".png")[0] + ".jpg", "jpeg")
        os.remove(filename)


def write_txt_files_yolo(yolo, f_train, f_val):
    """
    Definition: Fill in a text file containing a list of all images in the
        training and validation sets.

    Parameters: yolo - path to yolo dataset directory (contains 'Dataset\image_2' and 'Dataset\image_2_test')
                f_train - file open for adding training examples
                f_val - file open for adding validation examples
    Returns: None
    """
    for filename in glob.glob(os.path.join(yolo + "Dataset/image_2/", "*.*")):
        f_train.write('%s\n' % (filename))
    for filename in glob.glob(os.path.join(yolo + "Dataset/image_2_test/", "*.*")):
        f_val.write('%s\n' % (filename))


def make_yolo_directories(yolo):
    """
    Definition: Make directories for yolo images and labels.
        Removes previously created yolo image and label directories.

    Parameters: yolo - path to yolo directory to be created
    Returns: None
    """
    if os.path.exists(yolo):
        if python_version == 3:
            prompt = input('Directory already exists. Overwrite? (yes, no): ')
        else:
            prompt = input('Directory already exists. Overwrite? (yes, no): ')
        if prompt == 'no':
            exit(0)
        shutil.rmtree(yolo)
    os.makedirs(yolo)
    os.makedirs(yolo + "Dataset")
    os.makedirs(yolo + "Dataset/image_2")
    os.makedirs(yolo + "Dataset/label_2")
    # os.makedirs(yolo + "val")
    os.makedirs(yolo + "Dataset/image_2_test")
    os.makedirs(yolo + "Dataset/label_2_test")


def yolo(kitti_dir, yolo_dir, label=None):
    print("Converting kitti to yolo")

    # Split label file
    label_file = open(label)
    labels_split = label_file.read().split('\n')

    # Make all directories for yolo dataset
    make_yolo_directories(yolo_dir)

    # Iterate through kitti training data
    for f in os.listdir(kitti_dir + "Dataset/label_2"):
        fname = (kitti_dir + "Dataset/image_2/" + f).split(".txt")[0] + ".png"
        if os.path.isfile(fname):
            img = Image.open(fname)
            w, h = img.size
            img.close()
            labels, coords = parse_labels_yolo(os.path.join(kitti_dir +
                                                            "Dataset/label_2" + f), labels_split, w, h)
            yolof = open(yolo_dir + "Dataset/label_2" + f, "a+")
            for l, c in zip(labels, coords):
                yolof.write(l + " " + str(c[0]) + " " + str(c[1]) +
                            " " + str(c[2]) + " " + str(c[3]) + "\n")
            yolof.close()

    # Iterate through kitti validation data
    for f in os.listdir(kitti_dir + "Dataset/label_2_test"):
        fname = (kitti_dir + "val/images/" + f).split(".txt")[0] + ".png"
        if os.path.isfile(fname):
            img = Image.open(fname)
            w, h = img.size
            img.close()
            labels, coords = parse_labels_yolo(os.path.join(kitti_dir +
                                                            "Dataset/label_2/" + f), labels_split, w, h)
            yolof = open(yolo_dir + "Dataset/label_2/" + f, "a+")
            for l, c in zip(labels, coords):
                yolof.write(l + " " + str(c[0]) + " " + str(c[1]) +
                            " " + str(c[2]) + " " + str(c[3]) + "\n")
            yolof.close()

    # Copy images from kitti to yolo
    copy_images_yolo(kitti_dir, yolo_dir)

    # Create train.txt and val.txt and populate them
    f_train = open(yolo_dir + "train.txt", "a")
    f_val = open(yolo_dir + "val.txt", "a")
    write_txt_files_yolo(yolo_dir, f_train, f_val)
    f_train.close()
    f_val.close()


###########################################################
##########        KITTI to VOC Conversion        ##########
###########################################################
def write_voc_file(fname, labels, coords, img_width, img_height):
    """
    Definition: Writes label into VOC (XML) format.

    Parameters: fname - full file path to label file
                labels - list of objects in file
                coords - list of position of objects in file
                img_width - width of image
                img_height - height of image
    Returns: annotation - XML tree for image file
    """
    annotation = etree.Element('annotation')
    filename = etree.Element('filename')
    f = fname.split("/")
    filename.text = f[-1]
    annotation.append(filename)
    folder = etree.Element('folder')
    folder.text = "/".join(f[:-1])
    annotation.append(folder)
    for i in range(len(coords)):
        object = etree.Element('object')
        annotation.append(object)
        name = etree.Element('name')
        name.text = labels[i]
        object.append(name)
        bndbox = etree.Element('bndbox')
        object.append(bndbox)
        xmax = etree.Element('xmax')
        xmax.text = str(coords[i][2])
        bndbox.append(xmax)
        xmin = etree.Element('xmin')
        xmin.text = str(coords[i][0])
        bndbox.append(xmin)
        ymax = etree.Element('ymax')
        ymax.text = str(coords[i][3])
        bndbox.append(ymax)
        ymin = etree.Element('ymin')
        ymin.text = str(coords[i][1])
        bndbox.append(ymin)
        difficult = etree.Element('difficult')
        difficult.text = '0'
        object.append(difficult)
        occluded = etree.Element('occluded')
        occluded.text = '0'
        object.append(occluded)
        pose = etree.Element('pose')
        pose.text = 'Unspecified'
        object.append(pose)
        truncated = etree.Element('truncated')
        truncated.text = '1'
        object.append(truncated)
    img_size = etree.Element('size')
    annotation.append(img_size)
    depth = etree.Element('depth')
    depth.text = '3'
    img_size.append(depth)
    height = etree.Element('height')
    height.text = str(img_height)
    img_size.append(height)
    width = etree.Element('width')
    width.text = str(img_width)
    img_size.append(width)

    return annotation


def parse_labels_voc(label_file):
    """
    Definition: Parses label file to extract label and bounding box
        coordintates.

    Parameters: label_file - list of labels in images
    Returns: all_labels - contains a list of labels for objects in the image
             all_coords - contains a list of coordinates for objects in image
    """
    lfile = open(label_file)
    coords = []
    all_coords = []
    all_labels = []
    for line in lfile:
        l = line.split(" ")
        all_labels.append(l[0])
        coords = list(map(int, list(map(float, l[4:8]))))
        xmin = coords[0]
        ymin = coords[1]
        xmax = coords[2]
        ymax = coords[3]
        tmp = [xmin, ymin, xmax, ymax]
        all_coords.append(list(map(int, tmp)))
    lfile.close()
    return all_labels, all_coords


def copy_images_voc(kitti, voc):
    """
    Definition: Copy all images from the training and validation sets
        in kitti format to training and validation image sets in voc
        format.

    Parameters: kitti - path to kitti directory (contains 'train' and 'val')
                voc - path to voc output directory
    Returns: None
    """
    for filename in glob.glob(os.path.join(kitti + "Dataset/image_2/", "*.*")):
        shutil.copy(filename, voc + "Dataset/image_2/")
    for filename in glob.glob(os.path.join(kitti + "Dataset/image_2_test/", "*.*")):
        shutil.copy(filename, voc + "Dataset/image_2_test/")


def make_voc_directories(voc):
    """
    Definition: Make directories for voc images and labels.
        Removes previously created voc image and label directories.

    Parameters: yolo - path to voc directory to be created
    Returns: None
    """
    if os.path.exists(voc):
        if python_version == 3:
            prompt = input('Directory already exists. Overwrite? (yes, no): ')
        else:
            prompt = input('Directory already exists. Overwrite? (yes, no): ')
        if prompt == 'no':
            exit(0)
        shutil.rmtree(voc)
    os.makedirs(voc)
    os.makedirs(voc + "Dataset")
    os.makedirs(voc + "Dataset/image_2")
    os.makedirs(voc + "Dataset/label_2")
    # os.makedirs(yolo + "val")
    os.makedirs(voc + "Dataset/image_2_test")
    os.makedirs(voc + "Dataset/label_2_test")


def voc(kitti_dir, voc_dir, label=None):
    print("Convert kitti to voc")

    # Make all directories for voc dataset
    make_voc_directories(voc_dir)

    # Iterate through kitti training data
    for f in os.listdir(kitti_dir + "Dataset/label_2/"):
        fname = (kitti_dir + "Dataset/image_2/" + f).split(".txt")[0] + ".png"
        if os.path.isfile(fname):
            img = Image.open(fname)
            w, h = img.size
            img.close()
            labels, coords = parse_labels_voc(os.path.join(kitti_dir +
                                                           "Dataset/label_2/" + f))
            annotation = write_voc_file(fname, labels, coords, w, h)
            et = etree.ElementTree(annotation)
            et.write(voc_dir + "Dataset/label_2/" + f.split(".txt")[0] + ".xml", pretty_print=True)

    # Iterate through kitti validation data
    for f in os.listdir(kitti_dir + "Dataset/label_2_test"):
        fname = (kitti_dir + "Dataset/image_2_test" + f).split(".txt")[0] + ".png"
        if os.path.isfile(fname):
            img = Image.open(fname)
            w, h = img.size
            img.close()
            labels, coords = parse_labels_voc(os.path.join(kitti_dir +
                                                           "Dataset/label_2_test/" + f))
            annotation = write_voc_file(fname, labels, coords, w, h)
            et = etree.ElementTree(annotation)
            et.write(voc_dir + "Dataset/label_2_test/" + f.split(".txt")[0] + ".xml", pretty_print=True)

    # Copy images from kitti to voc
    copy_images_voc(kitti_dir, voc_dir)


###########################################################
##########        KITTI to LISA Conversion        #########
###########################################################
def lisa(kitti_dir, output, label=None):
    print("Convert kitti to lisa")
    pass

if __name__ == '__main__':
    voc("","Voc")
