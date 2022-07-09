from tensorflow import keras

def type_a(model_width):
    features = keras.Input(shape=(model_width*2+1, 6), name="note_pairs")
    distance = keras.Input(shape=(1,), name='dist')
    x = keras.layers.Conv1D(filters=32, kernel_size=32, activation='relu')(features)
    x = keras.layers.Conv1D(filters=32, kernel_size=16, activation='relu')(x)
    x = keras.layers.Conv1D(filters=64, kernel_size=16, activation='relu')(x)
    x = keras.layers.Conv1D(filters=64, kernel_size=8, activation='relu')(x)
    x = keras.layers.Conv1D(filters=128, kernel_size=8, activation='relu')(x)
    x = keras.layers.Conv1D(filters=128, kernel_size=4, activation='relu')(x)
    x = keras.layers.Conv1D(filters=1, kernel_size=4)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.concatenate([distance, x])
    x = keras.layers.Dense(units=128, activation='relu')(x)
    outputs = keras.layers.Dense(units=2, activation='softmax')(x)
    model = keras.Model(inputs=[features, distance], outputs=outputs)
    opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

    model.compile(optimizer=opt,
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[keras.metrics.SparseCategoricalAccuracy()])
    return model


def type_b(model_width):
    features = keras.Input(shape=(model_width*2+1, 6), name="note_pairs")
    distance = keras.Input(shape=(1,), name='dist')
    x = keras.layers.Conv1D(filters=32, kernel_size=32, activation='relu')(features)
    x = keras.layers.Conv1D(filters=32, kernel_size=16, activation='relu')(x)
    x = keras.layers.Conv1D(filters=64, kernel_size=16, activation='relu')(x)
    x = keras.layers.Conv1D(filters=64, kernel_size=8, activation='relu')(x)
    x = keras.layers.Conv1D(filters=128, kernel_size=8, activation='relu')(x)
    x = keras.layers.Conv1D(filters=128, kernel_size=4, activation='relu')(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.concatenate([distance, x])
    outputs = keras.layers.Dense(units=2, activation='softmax')(x)
    model = keras.Model(inputs=[features, distance], outputs=outputs)
    opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

    model.compile(optimizer=opt,
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[keras.metrics.SparseCategoricalAccuracy()])
    return model

# def type_c(model_width):
#     features = keras.Input(shape=(model_width, 3, 1), name="midi")
#     masks = keras.Input(shape=(model_width, 3, 1), name='mask')
#     x = keras.layers.concatenate([features, masks], axis=3)
#     x = keras.layers.Conv2D(filters=32, kernel_size=(32, 3), activation='relu')(x)
#     x = keras.layers.Conv2D(filters=32, kernel_size=(16, 1), activation='relu')(x)
#     x = keras.layers.Conv2D(filters=64, kernel_size=(16, 1), activation='relu')(x)
#     x = keras.layers.Conv2D(filters=64, kernel_size=(8, 1), activation='relu')(x)
#     x = keras.layers.Conv2D(filters=128, kernel_size=(8, 1), activation='relu')(x)
#     x = keras.layers.Conv2D(filters=128, kernel_size=(4, 1), activation='relu')(x)
#     x = keras.layers.Flatten()(x)
#     outputs = keras.layers.Dense(units=2, activation='softmax')(x)
#     model = keras.Model(inputs=[features, masks], outputs=outputs)
#     opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

#     model.compile(optimizer=opt,
#                   loss=keras.losses.SparseCategoricalCrossentropy(),
#                   metrics=[keras.metrics.SparseCategoricalAccuracy()])
#     return model

# def type_d(model_width):
#     features = keras.Input(shape=(model_width, 3), name="midi")
#     masks = keras.Input(shape=(model_width, 3), name='mask')
#     x = keras.layers.concatenate([features, masks], axis=2)
#     x = keras.layers.Conv1D(filters=32, kernel_size=32, activation='relu')(x)
#     x = keras.layers.Conv1D(filters=32, kernel_size=32, activation='relu')(x)
#     x = keras.layers.Conv1D(filters=64, kernel_size=16, activation='relu')(x)
#     x = keras.layers.Conv1D(filters=64, kernel_size=8, activation='relu')(x)
#     x = keras.layers.Conv1D(filters=128, kernel_size=8, activation='relu')(x)
#     x = keras.layers.Conv1D(filters=128, kernel_size=4, activation='relu')(x)
#     x = keras.layers.Flatten()(x)
#     outputs = keras.layers.Dense(units=2, activation='softmax')(x)
#     model = keras.Model(inputs=[features, masks], outputs=outputs)
#     opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

#     model.compile(optimizer=opt,
#                   loss=keras.losses.SparseCategoricalCrossentropy(),
#                   metrics=[keras.metrics.SparseCategoricalAccuracy()])
#     return model

model_dict = {}
model_dict['type_a'] = type_a
model_dict['type_b'] = type_b
# model_dict['type_c'] = type_c
# model_dict['type_d'] = type_d