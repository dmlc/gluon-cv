"""Visualize image."""
import numpy as np
import mxnet as mx

def plot_image(img, ax=None, reverse_rgb=False):
    """Visualize image.

    Parameters
    ----------
    img : numpy.ndarray or mxnet.nd.NDArray
        Image with shape `H, W, 3`.
    ax : matplotlib axes, optional
        You can reuse previous axes if provided.
    reverse_rgb : bool, optional
        Reverse RGB<->BGR orders if `True`.

    Returns
    -------
    matplotlib axes
        The ploted axes.

    Examples
    --------

    from matplotlib import pyplot as plt
    ax = plot_image(img)
    plt.show()
    """
    from matplotlib import pyplot as plt
    if ax is None:
        # create new axes
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    if isinstance(img, mx.nd.NDArray):
        img = img.asnumpy()
    img = img.copy()
    if reverse_rgb:
        img[:, :, (0, 1, 2)] = img[:, :, (2, 1, 0)]
    ax.imshow(img.astype(np.uint8))
    return ax

def cv_plot_image(img, scale=1, upperleft_txt=None, upperleft_txt_corner=(10, 100),
                  left_txt_list=None, left_txt_corner=(10, 150),
                  title_txt_list=None, title_txt_corner=(500, 50),
                  canvas_name='demo'):
    """Visualize image with OpenCV.

    Parameters
    ----------
    img : numpy.ndarray or mxnet.nd.NDArray
        Image with shape `H, W, 3`.
    scale : float
        The scaling factor of the output image
    upperleft_txt : str, optional, default is None
        If presents, will print the string at the upperleft corner
    upperleft_txt_corner : tuple, optional, default is (10, 100)
        The bottomleft corner of `upperleft_txt`
    left_txt_list : list of str, optional, default is None
        If presents, will print each string in the list close to the left
    left_txt_corner : tuple, optional, default is (10, 150)
        The bottomleft corner of `left_txt_list`
    title_txt_list : list of str, optional, default is None
        If presents, will print each string in the list close to the top
    title_txt_corner : tuple, optional, default is (500, 50)
        The bottomleft corner of `title_txt_list`
    canvas_name : str, optional, default is 'demo'
        The name of the canvas to plot the image

    Examples
    --------

    from matplotlib import pyplot as plt
    ax = plot_image(img)
    plt.show()
    """
    from ..filesystem import try_import_cv2
    cv2 = try_import_cv2()

    if isinstance(img, mx.nd.NDArray):
        img = img.asnumpy()

    height, width, _ = img.shape
    img = cv2.resize(img, (int(width * scale), int(height * scale)))
    if upperleft_txt is not None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = upperleft_txt_corner
        fontScale = 1
        fontColor = (255, 255, 255)
        thickness = 3

        cv2.putText(img, upperleft_txt, bottomLeftCornerOfText,
                    font, fontScale, fontColor, thickness)

    if left_txt_list is not None:
        starty = left_txt_corner[1]
        for txt in left_txt_list:
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (left_txt_corner[0], starty)
            fontScale = 1
            fontColor = (255, 255, 255)
            thickness = 1

            cv2.putText(img, txt, bottomLeftCornerOfText,
                        font, fontScale, fontColor, thickness)

            starty += 30

    if title_txt_list is not None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = title_txt_corner
        fontScale = 1
        fontColor = (255, 255, 255)
        thickness = 3

        for txt in title_txt_list:
            cv2.putText(img, txt, bottomLeftCornerOfText,
                        font, fontScale, fontColor, thickness)
            bottomLeftCornerOfText = (bottomLeftCornerOfText[0] + 100,
                                      bottomLeftCornerOfText[1] + 50)

    canvas = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow(canvas_name, canvas)
