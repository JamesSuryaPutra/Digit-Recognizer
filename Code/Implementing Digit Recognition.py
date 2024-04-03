# Preparation
# Import functions first
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

2024-03-06 09:56:59.349555: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2024-03-06 09:56:59.349655: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2024-03-06 09:56:59.474284: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered

# Import the competition data
# Import the competition data
train_data = pd.read_csv('../input/digit-recognizer/train.csv')
test_data = pd.read_csv('../input/digit-recognizer/test.csv')

print(train_data.shape)
print(test_data.shape)

(42000, 785)
(28000, 784)

# Display rows and columns
train_data.head(10)

label	pixel0	pixel1	pixel2	pixel3	pixel4	pixel5	pixel6	pixel7	pixel8	...	pixel774	pixel775	pixel776	pixel777	pixel778	pixel779	pixel780	pixel781	pixel782	pixel783
0	1	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
1	0	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
2	1	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
3	4	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
4	0	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
5	0	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
6	7	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
7	3	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
8	5	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
9	3	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0

10 rows x 785 columns

# Experimenting digits
# Drop labels
y = train_data['label']
train_data.drop("label", inplace=True, axis="columns")

# Plot the first five digits along with the matrix conversion
sample1 = np.reshape(train_data.iloc[2, :].values, (28, 28))
plt.imshow(sample1)
plt.show()

sample2 = np.reshape(train_data.iloc[3, :].values, (28, 28))
plt.imshow(sample2)
plt.show()

sample3 = np.reshape(train_data.iloc[4, :].values, (28, 28))
plt.imshow(sample3)
plt.show()

sample4 = np.reshape(train_data.iloc[5, :].values, (28, 28))
plt.imshow(sample4)
plt.show()

sample5 = np.reshape(train_data.iloc[6, :].values, (28, 28))
plt.imshow(sample5)
plt.show()

# Preprocessing the data
X = train_data.values.reshape((len(train_data), 28, 28))
X_test = test_data.values.reshape((len(test_data), 28, 28))

X = X / 255.0
X_test = X_test / 255.0

X = np.expand_dims(X, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, random_state=42)

print(f"No of data train: {X_train.shape[0]}")
print(f"No of data val: {X_val.shape[0]}")
print(f"No of data test: {X_test.shape[0]}")

No of data train: 33600
No of data val: 8400
No of data test: 28000

# Build and train the model
def build_model(input_shape=(28, 28, 1)):
    model = tf.keras.Sequential(
            layers=[tf.keras.layers.Input(shape=input_shape),
                    tf.keras.layers.Conv2D(32, 3, activation='relu', data_format="channels_last"),
                    tf.keras.layers.Conv2D(32, 3, activation='relu', data_format="channels_last"),
                    tf.keras.layers.MaxPooling2D(),
                    tf.keras.layers.Dropout(0.25),
                    tf.keras.layers.Conv2D(32, 3, activation='relu', data_format="channels_last"),
                    tf.keras.layers.Conv2D(32, 3, activation='relu', data_format="channels_last"),
                    tf.keras.layers.MaxPooling2D(),
                    tf.keras.layers.Dropout(0.25),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(32, activation='relu'),
                    tf.keras.layers.Dense(64, activation='relu'),
                    tf.keras.layers.Dropout(0.25),
                    tf.keras.layers.Dense(128, activation='relu'),
                    tf.keras.layers.Dense(10, activation='softmax'),
                   ]
            )

    model.build()
    model.summary()
    
    return model

model = build_model()

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss="sparse_categorical_crossentropy",
              metrics=["acc"])

callbacks = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_acc",
                                                 factor=0.5,
                                                 patience=5,
                                                 min_lr=1e-3)

hist = model.fit(X_train, y_train, validation_data = (X_val, y_val),
                 epochs=20,
                 callbacks=[callbacks])

Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                 │ (None, 26, 26, 32)     │           320 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_1 (Conv2D)               │ (None, 24, 24, 32)     │         9,248 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d (MaxPooling2D)    │ (None, 12, 12, 32)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (Dropout)               │ (None, 12, 12, 32)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_2 (Conv2D)               │ (None, 10, 10, 32)     │         9,248 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_3 (Conv2D)               │ (None, 8, 8, 32)       │         9,248 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_1 (MaxPooling2D)  │ (None, 4, 4, 32)       │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_1 (Dropout)             │ (None, 4, 4, 32)       │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten (Flatten)               │ (None, 512)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 32)             │        16,416 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 64)             │         2,112 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_2 (Dropout)             │ (None, 64)             │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (Dense)                 │ (None, 128)            │         8,320 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_3 (Dense)                 │ (None, 10)             │         1,290 │
└─────────────────────────────────┴────────────────────────┴───────────────┘

Total params: 56,202 (219.54 KB)
Trainable params: 56,202 (219.54 KB)
Non-trainable params: 0 (0.00 B)

Epoch 1/20
2024-03-06 09:57:23.380278: E external/local_xla/xla/service/gpu/buffer_comparator.cc:1137] Difference at 0: 2.35363, expected 1.53764
2024-03-06 09:57:23.380327: E external/local_xla/xla/service/gpu/buffer_comparator.cc:1137] Difference at 1: 1.82107, expected 1.00508
2024-03-06 09:57:23.380337: E external/local_xla/xla/service/gpu/buffer_comparator.cc:1137] Difference at 2: 1.81936, expected 1.00338
2024-03-06 09:57:23.380344: E external/local_xla/xla/service/gpu/buffer_comparator.cc:1137] Difference at 3: 2.23827, expected 1.42229
2024-03-06 09:57:23.380352: E external/local_xla/xla/service/gpu/buffer_comparator.cc:1137] Difference at 4: 3.05481, expected 2.23883
2024-03-06 09:57:23.380360: E external/local_xla/xla/service/gpu/buffer_comparator.cc:1137] Difference at 5: 2.58554, expected 1.76955
2024-03-06 09:57:23.380368: E external/local_xla/xla/service/gpu/buffer_comparator.cc:1137] Difference at 6: 3.00002, expected 2.18403
2024-03-06 09:57:23.380376: E external/local_xla/xla/service/gpu/buffer_comparator.cc:1137] Difference at 7: 2.82235, expected 2.00636
2024-03-06 09:57:23.380383: E external/local_xla/xla/service/gpu/buffer_comparator.cc:1137] Difference at 8: 3.10502, expected 2.28904
2024-03-06 09:57:23.380391: E external/local_xla/xla/service/gpu/buffer_comparator.cc:1137] Difference at 9: 2.73743, expected 1.92145
2024-03-06 09:57:23.380707: E external/local_xla/xla/service/gpu/conv_algorithm_picker.cc:705] Results mismatch between different convolution algorithms. This is likely a bug/unexpected loss of precision in cudnn.
(f32[32,32,26,26]{3,2,1,0}, u8[0]{0}) custom-call(f32[32,1,28,28]{3,2,1,0}, f32[32,1,3,3]{3,2,1,0}, f32[32]{0}), window={size=3x3}, dim_labels=bf01_oi01->bf01, custom_call_target="__cudnn$convBiasActivationForward", backend_config={"conv_result_scale":1,"activation_mode":"kRelu","side_input_scale":0,"leakyrelu_alpha":0} for eng20{k2=2,k4=1,k5=1,k6=0,k7=0} vs eng15{k5=1,k6=0,k7=1,k10=1}
2024-03-06 09:57:23.380745: E external/local_xla/xla/service/gpu/conv_algorithm_picker.cc:270] Device: Tesla P100-PCIE-16GB
2024-03-06 09:57:23.380758: E external/local_xla/xla/service/gpu/conv_algorithm_picker.cc:271] Platform: Compute Capability 6.0
2024-03-06 09:57:23.380770: E external/local_xla/xla/service/gpu/conv_algorithm_picker.cc:272] Driver: 12020 (535.129.3)
2024-03-06 09:57:23.380782: E external/local_xla/xla/service/gpu/conv_algorithm_picker.cc:273] Runtime: <undefined>
2024-03-06 09:57:23.380799: E external/local_xla/xla/service/gpu/conv_algorithm_picker.cc:280] cudnn version: 8.9.0
2024-03-06 09:57:23.494089: E external/local_xla/xla/service/gpu/buffer_comparator.cc:1137] Difference at 0: 2.35363, expected 1.53764
2024-03-06 09:57:23.494147: E external/local_xla/xla/service/gpu/buffer_comparator.cc:1137] Difference at 1: 1.82107, expected 1.00508
2024-03-06 09:57:23.494157: E external/local_xla/xla/service/gpu/buffer_comparator.cc:1137] Difference at 2: 1.81936, expected 1.00338
2024-03-06 09:57:23.494165: E external/local_xla/xla/service/gpu/buffer_comparator.cc:1137] Difference at 3: 2.23827, expected 1.42229
2024-03-06 09:57:23.494173: E external/local_xla/xla/service/gpu/buffer_comparator.cc:1137] Difference at 4: 3.05481, expected 2.23883
2024-03-06 09:57:23.494181: E external/local_xla/xla/service/gpu/buffer_comparator.cc:1137] Difference at 5: 2.58554, expected 1.76955
2024-03-06 09:57:23.494203: E external/local_xla/xla/service/gpu/buffer_comparator.cc:1137] Difference at 6: 3.00002, expected 2.18403
2024-03-06 09:57:23.494211: E external/local_xla/xla/service/gpu/buffer_comparator.cc:1137] Difference at 7: 2.82235, expected 2.00636
2024-03-06 09:57:23.494221: E external/local_xla/xla/service/gpu/buffer_comparator.cc:1137] Difference at 8: 3.10502, expected 2.28904
2024-03-06 09:57:23.494234: E external/local_xla/xla/service/gpu/buffer_comparator.cc:1137] Difference at 9: 2.73743, expected 1.92145
2024-03-06 09:57:23.494262: E external/local_xla/xla/service/gpu/conv_algorithm_picker.cc:705] Results mismatch between different convolution algorithms. This is likely a bug/unexpected loss of precision in cudnn.
(f32[32,32,26,26]{3,2,1,0}, u8[0]{0}) custom-call(f32[32,1,28,28]{3,2,1,0}, f32[32,1,3,3]{3,2,1,0}, f32[32]{0}), window={size=3x3}, dim_labels=bf01_oi01->bf01, custom_call_target="__cudnn$convBiasActivationForward", backend_config={"conv_result_scale":1,"activation_mode":"kRelu","side_input_scale":0,"leakyrelu_alpha":0} for eng20{k2=2,k4=1,k5=1,k6=0,k7=0} vs eng15{k5=1,k6=0,k7=1,k10=1}
2024-03-06 09:57:23.494282: E external/local_xla/xla/service/gpu/conv_algorithm_picker.cc:270] Device: Tesla P100-PCIE-16GB
2024-03-06 09:57:23.494295: E external/local_xla/xla/service/gpu/conv_algorithm_picker.cc:271] Platform: Compute Capability 6.0
2024-03-06 09:57:23.494305: E external/local_xla/xla/service/gpu/conv_algorithm_picker.cc:272] Driver: 12020 (535.129.3)
2024-03-06 09:57:23.494317: E external/local_xla/xla/service/gpu/conv_algorithm_picker.cc:273] Runtime: <undefined>
2024-03-06 09:57:23.494337: E external/local_xla/xla/service/gpu/conv_algorithm_picker.cc:280] cudnn version: 8.9.0

  56/1050 ━━━━━━━━━━━━━━━━━━━━ 2s 3ms/step - acc: 0.1580 - loss: 2.2034

WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1709719047.494494      69 device_compiler.h:186] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
                                                                                                                          
1045/1050 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - acc: 0.7367 - loss: 0.7591

2024-03-06 09:57:31.533219: E external/local_xla/xla/service/gpu/buffer_comparator.cc:1137] Difference at 0: 1.34972, expected 1.00161
2024-03-06 09:57:31.533265: E external/local_xla/xla/service/gpu/buffer_comparator.cc:1137] Difference at 1: 1.6774, expected 1.32929
2024-03-06 09:57:31.533275: E external/local_xla/xla/service/gpu/buffer_comparator.cc:1137] Difference at 2: 1.95868, expected 1.61057
2024-03-06 09:57:31.533283: E external/local_xla/xla/service/gpu/buffer_comparator.cc:1137] Difference at 3: 1.69333, expected 1.34521
2024-03-06 09:57:31.533291: E external/local_xla/xla/service/gpu/buffer_comparator.cc:1137] Difference at 4: 2.08558, expected 1.73747
2024-03-06 09:57:31.533298: E external/local_xla/xla/service/gpu/buffer_comparator.cc:1137] Difference at 6: 1.59326, expected 1.24515
2024-03-06 09:57:31.533306: E external/local_xla/xla/service/gpu/buffer_comparator.cc:1137] Difference at 8: 1.72639, expected 1.37827
2024-03-06 09:57:31.533314: E external/local_xla/xla/service/gpu/buffer_comparator.cc:1137] Difference at 9: 1.78453, expected 1.43641
2024-03-06 09:57:31.533322: E external/local_xla/xla/service/gpu/buffer_comparator.cc:1137] Difference at 10: 1.51148, expected 1.16336
2024-03-06 09:57:31.533329: E external/local_xla/xla/service/gpu/buffer_comparator.cc:1137] Difference at 11: 2.12228, expected 1.77416
2024-03-06 09:57:31.533343: E external/local_xla/xla/service/gpu/conv_algorithm_picker.cc:705] Results mismatch between different convolution algorithms. This is likely a bug/unexpected loss of precision in cudnn.
(f32[16,32,26,26]{3,2,1,0}, u8[0]{0}) custom-call(f32[16,1,28,28]{3,2,1,0}, f32[32,1,3,3]{3,2,1,0}, f32[32]{0}), window={size=3x3}, dim_labels=bf01_oi01->bf01, custom_call_target="__cudnn$convBiasActivationForward", backend_config={"conv_result_scale":1,"activation_mode":"kRelu","side_input_scale":0,"leakyrelu_alpha":0} for eng20{k2=2,k4=1,k5=1,k6=0,k7=0} vs eng15{k5=1,k6=0,k7=1,k10=1}
2024-03-06 09:57:31.533351: E external/local_xla/xla/service/gpu/conv_algorithm_picker.cc:270] Device: Tesla P100-PCIE-16GB
2024-03-06 09:57:31.533358: E external/local_xla/xla/service/gpu/conv_algorithm_picker.cc:271] Platform: Compute Capability 6.0
2024-03-06 09:57:31.533365: E external/local_xla/xla/service/gpu/conv_algorithm_picker.cc:272] Driver: 12020 (535.129.3)
2024-03-06 09:57:31.533372: E external/local_xla/xla/service/gpu/conv_algorithm_picker.cc:273] Runtime: <undefined>
2024-03-06 09:57:31.533382: E external/local_xla/xla/service/gpu/conv_algorithm_picker.cc:280] cudnn version: 8.9.0
2024-03-06 09:57:31.562470: E external/local_xla/xla/service/gpu/buffer_comparator.cc:1137] Difference at 0: 1.34972, expected 1.00161
2024-03-06 09:57:31.562504: E external/local_xla/xla/service/gpu/buffer_comparator.cc:1137] Difference at 1: 1.6774, expected 1.32929
2024-03-06 09:57:31.562513: E external/local_xla/xla/service/gpu/buffer_comparator.cc:1137] Difference at 2: 1.95868, expected 1.61057
2024-03-06 09:57:31.562521: E external/local_xla/xla/service/gpu/buffer_comparator.cc:1137] Difference at 3: 1.69333, expected 1.34521
2024-03-06 09:57:31.562529: E external/local_xla/xla/service/gpu/buffer_comparator.cc:1137] Difference at 4: 2.08558, expected 1.73747
2024-03-06 09:57:31.562537: E external/local_xla/xla/service/gpu/buffer_comparator.cc:1137] Difference at 6: 1.59326, expected 1.24515
2024-03-06 09:57:31.562546: E external/local_xla/xla/service/gpu/buffer_comparator.cc:1137] Difference at 8: 1.72639, expected 1.37827
2024-03-06 09:57:31.562558: E external/local_xla/xla/service/gpu/buffer_comparator.cc:1137] Difference at 9: 1.78453, expected 1.43641
2024-03-06 09:57:31.562571: E external/local_xla/xla/service/gpu/buffer_comparator.cc:1137] Difference at 10: 1.51148, expected 1.16336
2024-03-06 09:57:31.562592: E external/local_xla/xla/service/gpu/buffer_comparator.cc:1137] Difference at 11: 2.12228, expected 1.77416
2024-03-06 09:57:31.562615: E external/local_xla/xla/service/gpu/conv_algorithm_picker.cc:705] Results mismatch between different convolution algorithms. This is likely a bug/unexpected loss of precision in cudnn.
(f32[16,32,26,26]{3,2,1,0}, u8[0]{0}) custom-call(f32[16,1,28,28]{3,2,1,0}, f32[32,1,3,3]{3,2,1,0}, f32[32]{0}), window={size=3x3}, dim_labels=bf01_oi01->bf01, custom_call_target="__cudnn$convBiasActivationForward", backend_config={"conv_result_scale":1,"activation_mode":"kRelu","side_input_scale":0,"leakyrelu_alpha":0} for eng20{k2=2,k4=1,k5=1,k6=0,k7=0} vs eng15{k5=1,k6=0,k7=1,k10=1}
2024-03-06 09:57:31.562634: E external/local_xla/xla/service/gpu/conv_algorithm_picker.cc:270] Device: Tesla P100-PCIE-16GB
2024-03-06 09:57:31.562646: E external/local_xla/xla/service/gpu/conv_algorithm_picker.cc:271] Platform: Compute Capability 6.0
2024-03-06 09:57:31.562658: E external/local_xla/xla/service/gpu/conv_algorithm_picker.cc:272] Driver: 12020 (535.129.3)
2024-03-06 09:57:31.562670: E external/local_xla/xla/service/gpu/conv_algorithm_picker.cc:273] Runtime: <undefined>
2024-03-06 09:57:31.562685: E external/local_xla/xla/service/gpu/conv_algorithm_picker.cc:280] cudnn version: 8.9.0

1050/1050 ━━━━━━━━━━━━━━━━━━━━ 14s 4ms/step - acc: 0.7375 - loss: 0.7569 - val_acc: 0.9739 - val_loss: 0.0843 - learning_rate: 0.0010
Epoch 2/20
1050/1050 ━━━━━━━━━━━━━━━━━━━━ 3s 3ms/step - acc: 0.9623 - loss: 0.1298 - val_acc: 0.9780 - val_loss: 0.0688 - learning_rate: 0.0010
Epoch 3/20
1050/1050 ━━━━━━━━━━━━━━━━━━━━ 3s 3ms/step - acc: 0.9770 - loss: 0.0863 - val_acc: 0.9852 - val_loss: 0.0536 - learning_rate: 0.0010
Epoch 4/20
1050/1050 ━━━━━━━━━━━━━━━━━━━━ 3s 3ms/step - acc: 0.9771 - loss: 0.0865 - val_acc: 0.9798 - val_loss: 0.0621 - learning_rate: 0.0010
Epoch 5/20
1050/1050 ━━━━━━━━━━━━━━━━━━━━ 3s 3ms/step - acc: 0.9763 - loss: 0.0808 - val_acc: 0.9842 - val_loss: 0.0592 - learning_rate: 0.0010
Epoch 6/20
1050/1050 ━━━━━━━━━━━━━━━━━━━━ 3s 3ms/step - acc: 0.9796 - loss: 0.0754 - val_acc: 0.9885 - val_loss: 0.0432 - learning_rate: 0.0010
Epoch 7/20
1050/1050 ━━━━━━━━━━━━━━━━━━━━ 3s 3ms/step - acc: 0.9815 - loss: 0.0697 - val_acc: 0.9875 - val_loss: 0.0426 - learning_rate: 0.0010
Epoch 8/20
1050/1050 ━━━━━━━━━━━━━━━━━━━━ 3s 3ms/step - acc: 0.9831 - loss: 0.0648 - val_acc: 0.9869 - val_loss: 0.0519 - learning_rate: 0.0010
Epoch 9/20
1050/1050 ━━━━━━━━━━━━━━━━━━━━ 3s 3ms/step - acc: 0.9836 - loss: 0.0605 - val_acc: 0.9864 - val_loss: 0.0506 - learning_rate: 0.0010
Epoch 10/20
1050/1050 ━━━━━━━━━━━━━━━━━━━━ 3s 3ms/step - acc: 0.9840 - loss: 0.0626 - val_acc: 0.9888 - val_loss: 0.0385 - learning_rate: 0.0010
Epoch 11/20
1050/1050 ━━━━━━━━━━━━━━━━━━━━ 3s 3ms/step - acc: 0.9852 - loss: 0.0544 - val_acc: 0.9852 - val_loss: 0.0550 - learning_rate: 0.0010
Epoch 12/20
1050/1050 ━━━━━━━━━━━━━━━━━━━━ 3s 3ms/step - acc: 0.9826 - loss: 0.0692 - val_acc: 0.9888 - val_loss: 0.0438 - learning_rate: 0.0010
Epoch 13/20
1050/1050 ━━━━━━━━━━━━━━━━━━━━ 3s 3ms/step - acc: 0.9841 - loss: 0.0647 - val_acc: 0.9888 - val_loss: 0.0409 - learning_rate: 0.0010
Epoch 14/20
1050/1050 ━━━━━━━━━━━━━━━━━━━━ 3s 3ms/step - acc: 0.9838 - loss: 0.0631 - val_acc: 0.9896 - val_loss: 0.0421 - learning_rate: 0.0010
Epoch 15/20
1050/1050 ━━━━━━━━━━━━━━━━━━━━ 3s 3ms/step - acc: 0.9852 - loss: 0.0617 - val_acc: 0.9889 - val_loss: 0.0429 - learning_rate: 0.0010
Epoch 16/20
1050/1050 ━━━━━━━━━━━━━━━━━━━━ 3s 3ms/step - acc: 0.9870 - loss: 0.0563 - val_acc: 0.9874 - val_loss: 0.0494 - learning_rate: 0.0010
Epoch 17/20
1050/1050 ━━━━━━━━━━━━━━━━━━━━ 3s 3ms/step - acc: 0.9871 - loss: 0.0597 - val_acc: 0.9908 - val_loss: 0.0383 - learning_rate: 0.0010
Epoch 18/20
1050/1050 ━━━━━━━━━━━━━━━━━━━━ 3s 3ms/step - acc: 0.9863 - loss: 0.0570 - val_acc: 0.9864 - val_loss: 0.0587 - learning_rate: 0.0010
Epoch 19/20
1050/1050 ━━━━━━━━━━━━━━━━━━━━ 3s 3ms/step - acc: 0.9842 - loss: 0.0690 - val_acc: 0.9895 - val_loss: 0.0445 - learning_rate: 0.0010
Epoch 20/20
1050/1050 ━━━━━━━━━━━━━━━━━━━━ 3s 3ms/step - acc: 0.9881 - loss: 0.0533 - val_acc: 0.9887 - val_loss: 0.0463 - learning_rate: 0.0010

# Predict the model
pred = model.predict(X_test)
res = np.argmax(pred, axis=1)

875/875 ━━━━━━━━━━━━━━━━━━━━ 2s 1ms/step

# Save as CSV submission file
results = pd.Series(res, name="Label")
submission = pd.concat([pd.Series(range(1, X_test.shape[0] + 1), name="ImageId"), results], axis=1)
submission.to_csv("submission.csv", index=False)
print("Successfully saved as CSV file")

Successfully saved as CSV file
