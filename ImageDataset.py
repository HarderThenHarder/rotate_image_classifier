# !/usr/bin/env python3
"""
训练数据集类。

Author: pankeyu02
Date: 2022/05/18
"""
import math

import cv2
import torch
import numpy as np


class RotateImageDataset(torch.utils.data.Dataset):

    def __init__(self, input, input_shape=None, color_mode='rgb',
                    normalize=False, rotate=True, crop_center=True, 
                    crop_largest_rect=True):
        """
        初始化函数。

        Args:
            input (_type_): _description_
            input_shape (_type_, optional): _description_. Defaults to None.
            color_mode (str, optional): _description_. Defaults to 'rgb'.
            preprocess_func (_type_, optional): _description_. Defaults to None.
            rotate (bool, optional): _description_. Defaults to True.
            crop_cneter (bool, optional): _description_. Defaults to False.
            crop_largest_rect (bool, optional): _description_. Defaults to False.
        """
        self.images = None
        self.filenames = None
        self.input_shape = input_shape
        self.color_mode = color_mode
        self.rotate = rotate
        self.crop_center = crop_center
        self.crop_largest_rect = crop_largest_rect
        self.normalize = normalize

        if self.color_mode not in ['rgb', 'grayscale']:
            raise ValueError('Invalid color mode:', self.color_mode,
                                '; expected "rgb" or "grayscale".')

        # 检查输入类型，判断是数组类型还是文件列表
        if isinstance(input, (np.ndarray)):
            self.images = input
            self.N = self.images.shape[0]
            if not self.input_shape:
                self.input_shape = self.images.shape[1:]
                # add dimension if the images are greyscale
                if len(self.input_shape) == 2:
                    self.input_shape = (1,) + self.input_shape
        else:
            self.filenames = input
            self.N = len(self.filenames)
    
    def _largest_rotated_rect(self, w, h, angle):
        """
        Given a rectangle of size wxh that has been rotated by 'angle' (in
        radians), computes the width and height of the largest possible
        axis-aligned rectangle within the rotated rectangle.

        Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

        Converted to Python by Aaron Snoswell

        Source: http://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
        """

        quadrant = int(math.floor(angle / (math.pi / 2))) & 3
        sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
        alpha = (sign_alpha % math.pi + math.pi) % math.pi

        bb_w = w * math.cos(alpha) + h * math.sin(alpha)
        bb_h = w * math.sin(alpha) + h * math.cos(alpha)

        gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

        delta = math.pi - alpha - gamma

        length = h if (w < h) else w

        d = length * math.cos(alpha)
        a = d * math.sin(alpha) / math.sin(delta)

        y = a * math.cos(gamma)
        x = y * math.tan(gamma)

        return (
            bb_w - 2 * x,
            bb_h - 2 * y
        )


    def _crop_around_center(self, image, width, height):
        """
        Given a NumPy / OpenCV 2 image, crops it to the given width and height,
        around it's centre point

        Source: http://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
        """

        image_size = (image.shape[1], image.shape[0])
        image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

        if(width > image_size[0]):
            width = image_size[0]

        if(height > image_size[1]):
            height = image_size[1]

        x1 = int(image_center[0] - width * 0.5)
        x2 = int(image_center[0] + width * 0.5)
        y1 = int(image_center[1] - height * 0.5)
        y2 = int(image_center[1] + height * 0.5)

        return image[y1:y2, x1:x2]


    def _crop_largest_rectangle(self, image, angle, height, width):
        """
        Crop around the center the largest possible rectangle
        found with largest_rotated_rect.
        """
        return self._crop_around_center(
            image,
            *self._largest_rotated_rect(
                width,
                height,
                math.radians(angle)
            )
        )
    
    def _rotate(self, image, angle):
        """
        Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
        (in degrees). The returned image will be large enough to hold the entire
        new image, with a black background

        Source: http://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
        """
        # Get the image size
        # No that's not an error - NumPy stores image matricies backwards
        image_size = (image.shape[1], image.shape[0])
        image_center = tuple(np.array(image_size) / 2)

        # Convert the OpenCV 3x2 rotation matrix to 3x3
        rot_mat = np.vstack(
            [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
        )

        rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

        # Shorthand for below calcs
        image_w2 = image_size[0] * 0.5
        image_h2 = image_size[1] * 0.5

        # Obtain the rotated coordinates of the image corners
        rotated_coords = [
            (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
            (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
            (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
            (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
        ]

        # Find the size of the new image
        x_coords = [pt[0] for pt in rotated_coords]
        x_pos = [x for x in x_coords if x > 0]
        x_neg = [x for x in x_coords if x < 0]

        y_coords = [pt[1] for pt in rotated_coords]
        y_pos = [y for y in y_coords if y > 0]
        y_neg = [y for y in y_coords if y < 0]

        right_bound = max(x_pos)
        left_bound = min(x_neg)
        top_bound = max(y_pos)
        bot_bound = min(y_neg)

        new_w = int(abs(right_bound - left_bound))
        new_h = int(abs(top_bound - bot_bound))

        # We require a translation matrix to keep the image centred
        trans_mat = np.matrix([
            [1, 0, int(new_w * 0.5 - image_w2)],
            [0, 1, int(new_h * 0.5 - image_h2)],
            [0, 0, 1]
        ])

        # Compute the tranform for the combined rotation and translation
        affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

        # Apply the transform
        result = cv2.warpAffine(
            image,
            affine_mat,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR
        )

        return result
    
    def _generate_rotated_image(self, image, angle, size=None, crop_center=False,
                                    crop_largest_rect=False):
        """
        Generate a valid rotated image for the RotNetDataGenerator. If the
        image is rectangular, the crop_center option should be used to make
        it square. To crop out the black borders after rotation, use the
        crop_largest_rect option. To resize the final image, use the size
        option.
        """
        height, width = image.shape[:2]

        if crop_center:
            if width < height:
                height = width
            else:
                width = height

        image = self._rotate(image, angle)

        if crop_largest_rect:
            image = self._crop_largest_rectangle(image, angle, height, width)

        if size:
            image = cv2.resize(image, size)

        return image

    def _get_transformed_sample(self, index):
        """
        根据索引读取对应的数据，并返回旋转后的图片（X）和对应的旋转角度（label）。

        Args:
            index (_type_): 文件索引
        """
        if self.filenames is None:
            image = self.images[index]
        else:
            is_color = int(self.color_mode == 'rgb')                # 判断是否为灰度图
            img_name = self.filenames[index]
            image = cv2.imread(img_name, is_color)
            if is_color:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)      # 将opencv的默认BGR格式转为RGB格式
        
        if self.rotate:
            rotation_angle = np.random.randint(360)
        else:
            rotation_angle = 0

        # generate the rotated image
        rotated_image = self._generate_rotated_image(
            image,
            rotation_angle,
            size=self.input_shape[1:],
            crop_center=self.crop_center,
            crop_largest_rect=self.crop_largest_rect
        )

        # 如果是灰度图，则需要添加一个通道
        if rotated_image.ndim == 2:
            rotated_image = np.expand_dims(rotated_image, axis=2)
        
        # 归一化到[-1, 1]
        if self.normalize:
            rotated_image = (rotated_image / 255) * 2 - 1
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

            rotated_image[..., 0] -= mean[0]
            rotated_image[..., 1] -= mean[1]
            rotated_image[..., 2] -= mean[2]

            if std is not None:
                rotated_image[..., 0] /= std[0]
                rotated_image[..., 1] /= std[1]
                rotated_image[..., 2] /= std[2]
        
        if self.input_shape:
            rotated_image = np.reshape(rotated_image, self.input_shape)
        
        return rotated_image, rotation_angle

    def __len__(self):
        return self.N
    
    def __getitem__(self, idx):
        return self._get_transformed_sample(idx)


if __name__ == '__main__':
    import os
    import random

    from utils import get_filenames
    
    input_shape = (3, 244, 244)

    data_path = os.path.join('street_view')
    train_filenames, test_filenames = get_filenames(data_path)

    train_dataset = RotateImageDataset(input=train_filenames, input_shape=input_shape, normalize=False)

    # 随机生成旋转图片
    for _ in range(10):
        index = random.randint(1, 1000)
        img, target = train_dataset[index]
        img = np.reshape(img, (244, 244, 3)).astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite('./img_examples/angle_{}.png'.format(target), img)
