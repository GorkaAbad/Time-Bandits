import numpy as np
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from spikingjelly.datasets import play_frame
import os
import copy


data_dir = "data_augmented"
dataset = "gesture"
frames_number = 16

data_dir = os.path.join(data_dir, dataset)

transform = None

train_set = DVS128Gesture(
    data_dir,
    train=True,
    data_type="frame",
    split_by="number",
    frames_number=frames_number,
    transform=transform,
)
length = len(train_set)
print(f"Length of train set: {length}")

# increase the dataset in a 20% by rotating the images
# get 20% of the length of the dataset

for i in range(length):
    # remove all .npz that contain the word "rotated"
    img_path = train_set.samples[i][0]
    name = img_path.split("/")[-1].split(".")[0]
    path_without_name = "/".join(img_path.split("/")[:-1])
    extension = img_path.split("/")[-1].split(".")[1]
    print(name)
    if "rotated" in name or "center_crop" in name or "random_crop" in name:
        os.remove(img_path)
        print(f"Removed {img_path}")

# load again
train_set = DVS128Gesture(
    data_dir,
    train=True,
    data_type="frame",
    split_by="number",
    frames_number=frames_number,
    transform=transform,
)

length = len(train_set)
print(f"Length of train set: {length}")

augmented_length = int(length * 0.7)
idx = np.random.choice(length, augmented_length, replace=False)

for i in idx:
    image, label = train_set[i]
    img_path = train_set.samples[i][
        0
    ]  # data/gesture/frames_number_16_split_by_number/train/9/user23_led_0.npz

    name = img_path.split("/")[-1].split(".")[0]  # user23_led_0
    path_without_name = "/".join(
        img_path.split("/")[:-1]
    )  # data/gesture/frames_number_16_split_by_number/train/9

    image_copy = copy.deepcopy(image)

    # get a random angle to rotate the image
    angle = np.random.randint(0, 360)

    # play_frame(image_copy, save_gif_to=f"Before_rotation_{name}.gif")

    # rotate the image using np
    image_copy = np.rot90(
        image_copy, k=int(angle / 90), axes=(2, 3)
    ).copy()  # [T, 2, H, W]

    # play_frame(image_copy, save_gif_to=f"Rotated_{angle}_degrees.gif")

    # add the rotated image to the dataset as a tuple (image.npz, label)
    save_path = os.path.join(path_without_name, f"{name}_rotated_{angle}.npz")
    np.savez_compressed(save_path, frames=image_copy)

    # now consider doing center crop and random crop but ensure that the image size is the same as the original image. Use numpy to do this
    # center crop
    center_cropped_image = image_copy[:, :, 4:-4, 4:-4]  # crop 4 pixels from each side
    save_path_center_crop = os.path.join(
        path_without_name, f"{name}_rotated_{angle}_center_crop.npz"
    )

    # here the image is smaller than the original image so we add black padding to the image
    # the padding is 4 pixels on each side
    padding = 4
    center_cropped_image = np.pad(
        center_cropped_image,
        ((0, 0), (0, 0), (padding, padding), (padding, padding)),
        "constant",
        constant_values=0,
    )
    # play_frame(center_cropped_image, save_gif_to=f"Center_crop_{name}.gif")

    np.savez_compressed(save_path_center_crop, frames=center_cropped_image)
