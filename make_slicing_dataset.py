import os
import numpy as np
import nibabel as nib
import glob

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def window_CT(slice, min=-190, max=-30):
    """
    :param slice: 2D array as HxW having HU values in [-1024, +3071] = 2^12 bits
    :param min: minimum threshold value
    :param max: maximum threshold value
    :return: processed slice with values in [min, max] and then normalized to [0, 1]

    A) [-260, 340] HU: Best for visualization
    B) [+44] HU: Liver pixel values
    C) ...
    """
    sub = abs(min)
    diff = abs(min - max)

    img = slice + sub
    img[img <= 0] = 0  # min normalization
    img[img >= diff] = diff  # clipping prevents pixels from going beyond certain limits
    img = img / diff  # max normalization
    return img


def sliced_dataset(rawdataset_path, save_path, hu_list, organ_list):
    """
    :param rawdataset_path: 원본 Dataset 경로 ex) ./raw_dataset/
    :param save_path: Extracted sliced dataset folder name ^ save path ex) ./sliced_dataset/
    :param hu_list: Hounsfield List ex) [[-10,20],[20,50]]
    :param organ_list: Number of Organ, Reference https://flare22.grand-challenge.org/Home/
    :return: None
    """
    # Image, Label Data Path : rawdata 경로 지정
    imageFolder = sorted(glob.glob(f"{rawdataset_path}" + "images/*"))
    labelFolder = sorted(glob.glob(f"{rawdataset_path}" + "labels/*"))

    # Extracted Data Folder Name
    sliceFolderPath = save_path
    if os.path.exists(sliceFolderPath):
        print("already Slice DataSet exists")
        return
    else:
        os.mkdir(sliceFolderPath)

        sliceNum = []
        imgDataset = []
        labelDataset = []

        for img, lbl, idx in zip(imageFolder, labelFolder, range(len(imageFolder))):
            image = nib.load(img).get_fdata()
            label = nib.load(lbl).get_fdata()

            image = image.transpose(2, 1, 0)
            label = label.transpose(2, 1, 0)
            image = np.rot90(image, k=2, axes=(1, 2))
            label = np.rot90(label, k=2, axes=(1, 2))

            # Label 데이터 변환
            organ_num_list = organ_list
            extract_label = np.zeros(label.shape, dtype=np.uint8)
            for idx in range(len(organ_num_list)):
                indices = np.where(label == organ_num_list[idx])
                extract_label[indices] = idx + 1

            channel_img = []
            for min, max in hu_list:
                image_hu = np.asarray([window_CT(_, min, max) for _ in image])
                class_counts = np.sum(extract_label, axis=(1, 2))
                _img = []
                _lbl = []
                for i, count in enumerate(class_counts):
                    if count > 0:  # only non-zero slices
                        _img.append(image_hu[i])
                        _lbl.append(extract_label[i])  # only non-zero slices
                _img = np.asarray(_img)
                _lbl = np.asarray(_lbl)

                _img = (_img * 255).astype(np.uint8)  # [0.0, 1.0] --> [0, 255]
                _lbl = _lbl.astype(np.uint8)  # [0.0, 1.0] --> [0, 1] Caution: total foreground classes could be > 1!
                _lbl = np.squeeze(_lbl)[:, np.newaxis, :, :]
                _img = np.squeeze(_img)[:, np.newaxis, :, :]
                channel_img.append(_img)

            _channel_img = np.concatenate(channel_img, axis=1)

            imgDataset.append(_channel_img)
            labelDataset.append(_lbl)
            sliceNum.append(len(_img))

        imgDataset = np.concatenate(imgDataset, axis=0)
        labelDataset = np.concatenate(labelDataset, axis=0)

        np.savez_compressed(f"{sliceFolderPath}/concat_images.npz", data=imgDataset)
        np.savez_compressed(f"{sliceFolderPath}/concat_labels.npz", data=labelDataset)
        np.savez_compressed(f"{sliceFolderPath}/concat_idx.npz", data=np.array(sliceNum))

        print(f"Image Dataset Shape : {imgDataset.shape}")
        print(f"Label Dataset Shape : {labelDataset.shape}")
        print(f"The Number of Patience : {np.array(sliceNum).shape}")

if __name__ == '__main__':
    rawdata_path = 'data/raw_dataset/'
    savedata_path = 'data/test_dataset/'
    huList = [[10,20]]
    organNum = [2,3]
    sliced_dataset(rawdata_path, savedata_path, huList, organNum)
