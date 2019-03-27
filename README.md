# SE_ResNeXt_3D

The origin of [SENet](https://github.com/hujie-frank/SENet)

Code modified from 2d [se-resnext](https://github.com/taki0112/SENet-Tensorflow/blob/master/SE_ResNeXt.py)

## Note
The first layer was changed to ResNet-like structure due to the high memory cost.

- Before:
```python
  x = conv_layer(x, filter=64, kernel=[3, 3], stride=1, layer_name=scope+'_conv1')
  x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
```
- After:
```python
  x = Conv3D(filters=self.init_filters, kernel_size=(7, 7, 7), strides=(2, 2, 2), padding='same')(x)
  x = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding="same")(x)
  x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
```
