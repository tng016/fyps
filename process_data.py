import os
import numpy as np
import nibabel as nib
import imageio
from lxml import etree

def main():
    process_nii_image('Brats18_2013_4_1/Brats18_2013_4_1_t1.nii.gz')

def process_nii_image(example_filename):
    img_flair = nib.load(example_filename)
    img_data = img_flair.get_fdata()

    seg_flair = nib.load('Brats18_2013_4_1/Brats18_2013_4_1_seg.nii.gz')
    seg_data = seg_flair.get_fdata()

    train_text = ''
    val_text = ''
    tumor_text = ''
    for i in range(155):
        filename = 'Brats18_2013_4_1_t1_'+str(i)
        img_data_slice = img_data[:,:,i]
        seg_data_slice = seg_data[:,:,i]
        imageio.imwrite('./brats18/trainval/JPEGImages/Brats18_2013_4_1_t1_'+str(i)+'.jpg', img_data_slice)
        x0= first_nonzero(seg_data_slice, axis=0, invalid_val=9999)
        y0= first_nonzero(seg_data_slice, axis=1, invalid_val=9999)
        x1= last_nonzero(seg_data_slice, axis=0, invalid_val=-1)
        y1= last_nonzero(seg_data_slice, axis=1, invalid_val=-1)
        write_xml(filename,x0,x1,y0,y1)
        if (x0 != 9999 and y0 !=  9999 and x1 !=  -1 and y1 !=  -1):
            tumor_text += filename +' 1\n'
        else:
            tumor_text += filename +' -1\n'
        #if (i%5!=0):
        if True:
            train_text+=filename +'\n'
        else:
            val_text+=filename +'\n'
    with open("./brats18/trainval/ImageSets/Main/valid.txt", "w") as text_file:
        text_file.write(val_text)
    with open("./brats18/trainval/ImageSets/Main/train.txt", "w") as text_file:
        text_file.write(train_text)
    with open("./brats18/trainval/ImageSets/Main/tumor.txt", "w") as text_file:
        text_file.write(tumor_text)

def first_nonzero(arr, axis, invalid_val=9999):
    mask = arr!=0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val).min()

def last_nonzero(arr, axis, invalid_val=-1):
    mask = arr!=0
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val).max()

def write_xml(example_filename,x0,x1,y0,y1):
    # create XML
    root = etree.Element('annotation')

    branch = etree.Element('folder')
    branch.text = 'Brats2018'
    root.append(branch)

    branch = etree.Element('filename')
    branch.text = example_filename+'.jpg'
    root.append(branch)

    branch = etree.Element('source')
    secondary_branch = etree.Element('database')
    secondary_branch.text = 'The Brats 2018 databse'
    branch.append(secondary_branch)
    root.append(branch)

    branch = etree.Element('size')
    secondary_branch = etree.Element('width')
    secondary_branch.text = '155'
    branch.append(secondary_branch)
    secondary_branch = etree.Element('height')
    secondary_branch.text = '155'
    branch.append(secondary_branch)
    secondary_branch = etree.Element('depth')
    secondary_branch.text = '1'
    branch.append(secondary_branch)
    root.append(branch)

    if (x0 != 9999 and y0 !=  9999 and x1 !=  -1 and y1 !=  -1):
        branch = etree.Element('object')
        secondary_branch = etree.Element('name')
        secondary_branch.text = 'tumor'
        branch.append(secondary_branch)
        secondary_branch = etree.Element('pose')
        secondary_branch.text = 'Unspecified'
        branch.append(secondary_branch)
        secondary_branch = etree.Element('truncated')
        secondary_branch.text = '0'
        branch.append(secondary_branch)
        secondary_branch = etree.Element('difficult')
        secondary_branch.text = '0'
        branch.append(secondary_branch)

        secondary_branch = etree.Element('bndbox')
        third_branch = etree.Element('xmin')
        third_branch.text = str(x0)
        secondary_branch.append(third_branch)
        third_branch = etree.Element('ymin')
        third_branch.text = str(y0)
        secondary_branch.append(third_branch)
        third_branch = etree.Element('xmax')
        third_branch.text = str(x1)
        secondary_branch.append(third_branch)
        third_branch = etree.Element('ymax')
        third_branch.text = str(y1)
        secondary_branch.append(third_branch)
        branch.append(secondary_branch)
        root.append(branch)

    # write xml
    tree = etree.ElementTree(root)
    tree.write('./brats18/trainval/Annotations/'+example_filename+".xml")

if __name__ == "__main__": main()
