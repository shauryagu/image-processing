import os
import cv2
import numpy as np
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from util import generate_gif, renderCube


def rotX(theta):
    """
    Generate 3D rotation matrix about X-axis
    Input:  theta: rotation angle about X-axis
    Output: Rotation matrix (3 x 3 array)
    """
    return np.array([[1, 0, 0], [0, np.cos(theta), -1 * np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])



def rotY(theta):
    """
    Generate 3D rotation matrix about Y-axis
    Input:  theta: rotation angle along y-axis
    Output: Rotation matrix (3 x 3 array)
    """
    return np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-1*np.sin(theta), 0, np.cos(theta)]])


def part1():
    # TODO: Solution for Q1
    # Task 1: Use rotY() to generate cube.gif
    """
    R = np.empty((20,3,3), dtype=np.float64)
    count = 0
    for rot in np.linspace(0, 2*np.pi, num = 20):
        R[count] = rotY(rot)
        #R[count] = rotX(rot)
        count += 1
    generate_gif(R)
    """
    # Task 2:  Use rotX() and rotY() sequentially to check
    # the commutative property of Rotation Matrices
    #renderCube(R = np.dot(rotY(np.pi / 4), rotX(np.pi / 4)), file_name = 'yx.png')
    # Task 3: Combine rotX() and rotY() to render a cube 
    # projection such that end points of diagonal overlap
    #renderCube(R = np.dot(rotY(np.pi / 4), rotX(np.pi / 4)), file_name = 'single_point.png')
    #renderCube(R = np.dot(rotX(2*np.pi - .6155), rotY(np.pi + .785)), file_name='single_point.png')
    # Hint: Try rendering the cube with multiple configrations
    # to narrow down the search region
    pass


def split_triptych(trip):
    """
    Split a triptych into thirds
    Input:  trip: a triptych (H x W matrix)
    Output: R, G, B martices
    """
    R, G, B = None, None, None
    # TODO: Split a triptych into thirds and 
    image = plt.imread(trip);
    R = image[int(2*image.shape[0]/3):image.shape[0], :]
    print(R.shape)
    G = image[int(image.shape[0]/3):int(2*image.shape[0]/3), :]
    print(G.shape)
    B = image[0:int(image.shape[0]/3), :]
    print(B.shape)
    return R, G, B


def normalized_cross_correlation(ch1, ch2):
    """
    Calculates similarity between 2 color channels
    Input:  ch1: channel 1 matrix
            ch2: channel 2 matrix
    Output: normalized cross correlation (scalar)

    ch1 = ch1 - np.nanmean(ch1[10:(ch1.shape[0]-10), 10:(ch1.shape[1]-10)])
    ch2 = ch2 - np.nanmean(ch2[10:(ch2.shape[0]-10), 10:(ch2.shape[1]-10)])
    ch1 = ch1 / np.linalg.norm(ch1[10:(ch1.shape[0]-10), 10:(ch1.shape[1]-10)])
    ch2 = ch2 / np.linalg.norm(ch2[10:(ch2.shape[0]-10), 10:(ch2.shape[1]-10)])
    """
    ch1 = ch1 - np.nanmean(ch1)
    ch2 = ch2 - np.nanmean(ch2)
    ch1 = ch1 / np.linalg.norm(ch1)
    ch2 = ch2 / np.linalg.norm(ch2)

    #return np.sum(ch1[10:(ch1.shape[0]-10), 10:(ch1.shape[1]-10)]*ch2[10:(ch2.shape[0]-10), 10:(ch2.shape[1]-10)])
    return np.sum(ch1*ch2)



def best_offset(ch1, ch2, metric, Xrange=np.arange(-10, 10),
                Yrange=np.arange(-10,10)):
    """
    Input:  ch1: channel 1 matrix
            ch2: channel 2 matrix
            metric: similarity measure between two channels
            Xrange: range to search for optimal offset in vertical direction
            Yrange: range to search for optimal offset in horizontal direction
    Output: optimal offset for X axis and optimal offset for Y axis

    Note: Searching in Xrange would mean moving in the vertical 
    axis of the image/matrix, Yrange is the horizontal axis 
    """
    # TODO: Use metric to align ch2 to ch1 and return optimal offsets
    offset = np.empty(2)
    bestSim = 0
    currentSim = 0
    for x in Xrange:
        for y in Yrange:
            currentSim = normalized_cross_correlation(ch1, np.roll(ch2, (x,y), axis=(1, 0)))
            if currentSim > bestSim:
                bestSim = currentSim
                offset[0] = x
                offset[1] = y
    return offset


def align_and_combine(R, G, B, metric):
    """
    Input:  R: red channel
            G: green channel
            B: blue channel
            metric: similarity measure between two channels
    Output: aligned RGB image 
    """
    # TODO: Use metric to align the three channels 
    # Hint: Use one channel as the anchor to align other two
    o1 = best_offset(ch1=R, ch2=G, metric = normalized_cross_correlation)
    print("final")
    print(o1)
    o2 = best_offset(ch1=R, ch2=B, metric = normalized_cross_correlation)
    print("final")
    print(o2)
    G = np.roll(G, int(o1[0]), axis=1)
    G = np.roll(G, int(o1[1]), axis=0)
    B = np.roll(B, int(o2[0]), axis=1)
    B = np.roll(B, int(o2[1]), axis=0)

    return np.stack((R, G, B), axis=2)


def pyramid_align(ref, tar, level, off):
    # TODO: Reuse the functions from task 2 to perform the 
    # image pyramid alignment iteratively or recursively
    if level == 0:
        o = best_offset(ref, tar, metric = normalized_cross_correlation)
        print(level)
        print(o)
        return o
    offset = pyramid_align(cv2.resize(ref, (int(ref.shape[0]/4), int(ref.shape[1]/4))),
                           cv2.resize(tar, (int(tar.shape[0]/4), int(tar.shape[1]/4))), level-1)
    offset = offset * 4
    print(level)
    print(offset)
    tar = np.roll(tar, (int(offset[0]), int(offset[1])), axis=(1,0))

    return best_offset(ref, tar, metric=normalized_cross_correlation)






def part2():
    # TODO: Solution for Q2
    # Task 1: Generate a colour image by splitting 
    # the triptych image and save it
    R, G, B = split_triptych('/Users/shauryagunderia/downloads/eecs442/hw1/tableau/vancouver_tableau.jpg')
    #plt.imsave('color.jpg', np.stack((R,G,B), axis=2))
    # Task 2: Remove misalignment in the colour channels 
    # by calculating best offset
    #plt.imsave('color.jpg', align_and_combine(R=R, G=G, B=B, metric = normalized_cross_correlation))
    # Task 3: Pyramid alignment
    oG = pyramid_align(R, G, level=3)
    print(oG)
    oB = pyramid_align(R, B, level=3)
    print(oB)
    G = np.roll(G, (int(oG[0]), int(oG[1])), axis=(1,0))
    B = np.roll(B, (int(oB[0]), int(oB[1])), axis=(1,0))
    img = align_and_combine(R, G, B, metric=normalized_cross_correlation)
    plt.imsave('colorPY1.jpg', img)

    pass


def part3():
    # TODO: Solution for Q3
    img = plt.imread('/Users/shauryagunderia/downloads/eecs442/hw1/rubik/indoor.png')
    imgOut = plt.imread('/Users/shauryagunderia/downloads/eecs442/hw1/rubik/outdoor.png')
    #R = plt.imshow(img[:,:,0], cmap='gray')
    #plt.show()
    #G = plt.imshow(img[:,:,1], cmap='gray')
    #plt.show()
    #B = plt.imshow(img[:,:,2], cmap='gray')
    #plt.show()
    '''
    imgOut[:, :, 0] = imgOut[:, :, 0] / np.linalg.norm(imgOut[:, :, 0])
    imgOut[:, :, 1] = imgOut[:, :, 1] / np.linalg.norm(imgOut[:, :, 1])
    imgOut[:, :, 2] = imgOut[:, :, 2] / np.linalg.norm(imgOut[:, :, 2])
    print(imgOut)
    img[:, :, 0] = img[:, :, 0] / np.linalg.norm(img[:, :, 0])
    img[:, :, 1] = img[:, :, 1] / np.linalg.norm(img[:, :, 1])
    img[:, :, 2] = img[:, :, 2] / np.linalg.norm(img[:, :, 2])
    print(img)
    '''

    imgOut = cv2.cvtColor(imgOut, cv2.COLOR_RGB2LAB)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    #imgOut = cv2.cvtColor(imgOut, cv2.COLOR_Lab2RGB)
    #img = cv2.cvtColor(img, cv2.COLOR_Lab2RGB)
    plt.imshow(imgOut[:,:,2], cmap='gray')
    plt.show()
    plt.imshow(img[:,:,2], cmap='gray')
    plt.show()
    '''
    im1 = plt.imread('/Users/shauryagunderia/downloads/eecs442/hw1/IMG_5652.JPG')
    im2 = plt.imread('/Users/shauryagunderia/downloads/eecs442/hw1/IMG_5657.JPG')
    plt.imshow(im1)
    plt.show()
    plt.imshow(im2)
    plt.show()
    '''

    pass
    

def main():
    #part1()
    part2()
    #part3()


if __name__ == "__main__":
    main()
