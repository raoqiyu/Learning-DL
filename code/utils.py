# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 20:53:34 2015

@author: wahah
"""

import numpy


def scale_to_unit_interval(ndar, eps=1e-8):
	"""Scale all values in the ndarray ndar to be between 0 and 1"""
	ndar = ndar.copy()
	ndar -= ndar.min()
	ndar *= 1.0/(ndar.max() + eps)
	return ndar

def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0,0),
                       scale_rowsto_unit_interval=True,output_pixel_vals=True):
    """
    Transform an array with one flattend image per row, into an array in which
    images are reshaped and layed out like tiles on floor.

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can be
                2-D ndarray or None
    :param X: a 2-D ndarray in which every row is a flattend image.

    :type  img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type  tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile

    :type  tile_spacing: tuple; (height, weidth)
    :param tile_spacing: the space between images on the directon of rows and
                         cols respectively

    :param scale_rowsto_unit_interval: if the value need to be scaled  to [0,1]
                        before being plottedor not

    :param output_pixel_vals: if output should be pixel values (i.e. int8 values)
                        or float

    :returns: aray suitable for viewing as an image.
    :return type: a 2-D array with same dtype as X.
    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2


    out_shape = [0,0]
    out_shape[0] = (img_shape[0] + tile_spacing[0]) * tile_shape[0] - tile_spacing[0]
    out_shape[1] = (img_shape[1] + tile_spacing[1]) * tile_shape[1] - tile_spacing[1]

    # Dealing with color images,
    # input X has 4 dimension,(R, G, B, alpha)
    if isinstance(X, tuple):
        assert len(X) == 4
        # create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = numpy.zeros((out_shape[0],out_shape[1],4),dtype='uint8')
        else:
            out_array = numpy.zeros((out_shape[0],out_shape[1],4),dtype=X.dtype)

        # colors default to 0 (black), alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0, 0, 0, 1.]

        for i in range(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                # Using broadcast of numpy's array
                out_array[:, :, i] = numpy.zeros(
                    out_shape,dtype=dt
                ) + channel_defaults[i]
            else:
                # Using recurrent call to compute the channel and store it
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rowsto_unit_interval, output_pixel_vals
                )
            return out_array
    else:
        imgH, imgW = img_shape
        spacingH, spacingW = tile_spacing

        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = numpy.zeros(out_shape, dtype=dt)

        for tile_row in range(tile_shape[0]):
            for tile_col in range(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_X = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rowsto_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        this_img = scale_to_unit_interval(
                            this_X.reshape(img_shape)
                        )
                    else:
                        this_img = this_X.reshape(img_shape)
                    # add this image slice to the corresponding position
                    # in the out_array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (imgH + spacingH):tile_row *(imgH +spacingH)+imgH,
                        tile_col * (imgW + spacingW):tile_col *(imgW +spacingW)+imgW
                    ] = this_img * c
    return out_array
