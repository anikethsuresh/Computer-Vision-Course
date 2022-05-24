import cv2
import numpy as np
from enum import Enum

class Direction(Enum):
    """
    Direction Enum to specify coordinate direction
    """
    X = 1
    Y = 2

def get_pixel_val(image: np.ndarray, x:int, y:int):
    """
    Function to get the pixel value

    Args:
        image (np.ndarray): Input image
        x (int): x coordinate
        y (int): y coordinate

    Returns:
        value of pixel if x and y are within the image
        Or, 0, if x or y are outsize the image boundary 
    """
    if x < 0 or y < 0 or x >= image.shape[0] or y >= image.shape[1]:
        return 0
    else:
        return image[x][y]


def gradient(image: np.ndarray, direction: Direction, diff:int = 1):
    """
    Second order derivative of the image, in the direction specified

    Args:
        image (np.ndarray): input image
        direction (Direction): x or y direction
        diff (int, optional): Difference between pixels to calculate the derivative. Defaults to 1.

    Returns:
        image (np.ndarray): derivative of the image
    """
    image_grad = np.zeros(image.shape)
    for i in range(image.shape[0]): # rows
        for j in range(image.shape[1]): # columns
            if direction == Direction.X:
                prev = get_pixel_val(image, i, j - diff)
                next = get_pixel_val (image, i, j + diff)
            elif direction == Direction.Y:
                prev = get_pixel_val(image, i - diff, j)
                next = get_pixel_val (image, i + diff, j)
            image_grad[i][j] = (next + prev - 2*(get_pixel_val(image, i, j)))/(diff*diff)
    return image_grad


def zero_crossings(image: np.ndarray, direction: Direction):
    """
    Identify zero crossing in the direction specified

    Args:
        image (np.ndarray): input image
        direction (Direction): Direction

    Returns:
        image (np.ndarray): bool image with zero crossings where pixel = 1 
    """
    image_grad = np.zeros(image.shape)
    diff = 1
    for i in range(image.shape[0]): # rows
        for j in range(image.shape[1]): # columns
            if direction == Direction.X:
                prev = get_pixel_val(image, i, j - diff)
                next = get_pixel_val (image, i, j + diff)
            elif direction == Direction.Y:
                prev = get_pixel_val(image, i - diff, j)
                next = get_pixel_val (image, i + diff, j)
            if get_pixel_val(image, i, j) == 0:
                image_grad[i][j] = 1
            elif (prev > 0 and next <= 0) or (prev <0 and next >=0):
                image_grad[i][j] =1
    return image_grad


def main():
    IMAGE_NAME = "images/tools.png" # https://homepages.inf.ed.ac.uk/rbf/HIPR2/zeros.htm
    image = cv2.imread(IMAGE_NAME, cv2.IMREAD_GRAYSCALE)
    i = 1 # Distance between pixels
    grad_x = gradient(image, Direction.X, i) # Gradient in X
    grad_y = gradient(image, Direction.Y, i) # Gradient in Y
    grad = np.square(grad_x) + np.square(grad_y) # The laplacian
    grad = grad.astype(np.uint8)
    zero_crossing_x = zero_crossings(grad, Direction.X) # Identify zero crossing in X
    zero_crossing_y = zero_crossings(grad, Direction.Y) # Identify zero crossing in y
    zero_crossing = np.logical_or(zero_crossing_x, zero_crossing_y).astype(float) # Get locations where the z

    cv2.imshow("tools", image) # Original Image
    cv2.imshow("zero crossing",zero_crossing) # Zero crossings of Laplacian

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()