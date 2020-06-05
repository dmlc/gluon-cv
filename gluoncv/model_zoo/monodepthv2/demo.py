from mxnet import nd


def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return nd.UpSampling(x, scale=2, sample_type='bilinear')


if __name__ == '__main__':
    x = nd.ones((1, 512, 24 ,80))

    x = upsample(x)

    print(x.shape)