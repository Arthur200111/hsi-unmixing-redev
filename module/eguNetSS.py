from keras.initializers.initializers_v2 import GlorotUniform
from keras.layers import Input, Conv2D, BatchNormalization, Dropout, Activation, ReLU, Softmax, AveragePooling2D, Conv2DTranspose, Reshape
from keras.models import Model

from keras.optimizers import Adam

import numpy.typing as npt

from keras.utils.vis_utils import plot_model

from keras.losses import CategoricalCrossentropy, MeanSquaredError
from keras import regularizers

import numpy as np

class eguNetss:
    model: Model
    r: int
    n_pure: int
    size: tuple
    pure: npt.NDArray[np.float64]
    mixed: npt.NDArray[np.float64]
    abundance_pure: npt.NDArray[np.float64]
    abundance_mixed: npt.NDArray[np.float64] 
    
    def __init__(self, r, n_pure = 5, size = (16, 16), endmembers_count = 5):
        self.r = r
        self.n_pure = n_pure
        self.size = size
        self.endmembers_count = endmembers_count

    def initModel(self, isTraining=True, keep_prob=.5, momentum=.9, reg = 0.002):
        
        pure_input = Input(shape=(self.endmembers_count, 1, self.r), name="i_p")
        mixed_input = Input(shape=(self.size[0], self.size[1], self.r), name="i_m")
        
        initializer = GlorotUniform(seed=1)
        regularizer = regularizers.L2(reg)
        
        ## Layer 1
        pure = Conv2D(128, (1, 1), padding="same", kernel_initializer=initializer, name='L1_C_p', kernel_regularizer=regularizer)(pure_input)
        pure = BatchNormalization(momentum=momentum, axis=3, name='L1_BN_p')(pure, training=isTraining)
        pure = Dropout(1-keep_prob, name='L1_D_p')(pure)
        pure = Activation("tanh", name='L1_A_T_p')(pure)
        
        mixed = Conv2D(128, (5, 5), padding="same", kernel_initializer=initializer, name='L1_C_m', kernel_regularizer=regularizer)(mixed_input)
        mixed = BatchNormalization(momentum=momentum, axis=3, name='L1_BN_m')(mixed, training=isTraining)
        mixed = Dropout(1-keep_prob, name='L1_D_m')(mixed)
        # mixed = AveragePooling2D(2,2, name='L1_AP_m')(mixed)
        mixed = Activation("tanh", name = 'L1_A_T_m')(mixed)
        
        ## Layer 2
        pure = Conv2D(64, (1,1), padding="same", kernel_initializer=initializer, name='L2_C_p', kernel_regularizer=regularizer)(pure)
        pure = BatchNormalization(momentum=momentum, axis=3, name='L2_BN_p')(pure, training=isTraining)
        pure = Activation("tanh", name='L2_A_T_p')(pure)
        
        mixed = Conv2D(64, (3, 3), padding="same", kernel_initializer=initializer, name='L2_c_m', kernel_regularizer=regularizer)(mixed)
        mixed = BatchNormalization(momentum=momentum, axis=3, name='L2_BN_m')(mixed, training=isTraining)
        # mixed = AveragePooling2D(2, 2, name='L2_AP_m')(mixed)
        mixed = Activation("tanh", name='L2_A_T_m')(mixed)
        
        ## Layer 3
        pure = Reshape([self.endmembers_count, 1, 64], name='L3_R_p')(pure)
        mixed = Reshape([mixed.shape[1], mixed.shape[2], 64], name='L3_R_m')(mixed)
        
        sharedCov3 = Conv2D(32, (1,1), padding="same", kernel_initializer=initializer, name='L3_C_shared', kernel_regularizer=regularizer)
        
        pure = sharedCov3(pure)
        pure = BatchNormalization(momentum=momentum, axis=3, name='L3_BN_p')(pure, training=isTraining)
        pure = ReLU(name='L3_A_RL_p')(pure)
        
        mixed = sharedCov3(mixed)
        mixed = BatchNormalization(momentum=momentum, axis=3, name='L3_BN_m')(mixed, training=isTraining)
        # mixed = AveragePooling2D(2, 2, name='L3_AP_m')(mixed)
        mixed = ReLU(name='L3_A_RL_m')(mixed)
        
        ## Layer 4
        sharedCov4 = Conv2D(self.n_pure, (1,1), padding="same", kernel_initializer=initializer, name='L4_C_shared', kernel_regularizer=regularizer)
        
        # pure = Conv2D(self.n_pure, (1,1), padding="same", kernel_initializer=initializer, name='L4_C_p', kernel_regularizer=regularizer)(pure)
        pure = sharedCov4(pure)
        abundances_pure = Softmax(name='abundances_pure', axis=-1)(pure)
        # abundances_pure = Softmax(name='L3_A_SM_p')(pure)
        # abundances_pure = Reshape([-1], name = 'abundances_pure')(abundances_pure)
        
        # mixed = Conv2DTranspose(self.n_pure, (1,1), padding="same", kernel_initializer=initializer, name='L4_CT_m', kernel_regularizer=regularizer)(mixed)
        mixed = sharedCov4(mixed)
        abundances_mixed = Softmax(name='abundances_mixed', axis=-1)(mixed)
        
        ## Layer DE 1
        mixed = Conv2DTranspose(32, (1,1), padding="same", kernel_initializer=initializer, name='LD1_CT_m', kernel_regularizer=regularizer)(mixed)
        mixed = BatchNormalization(momentum=momentum, axis=3, name='LD1_BN_m')(mixed, training=isTraining)
        mixed = Activation("sigmoid", name='LD1_A_S_m')(mixed)
        
        ## Layer DE 2
        mixed = Conv2DTranspose(64, (1,1), padding="same", kernel_initializer=initializer, name='LD2_CT_m', kernel_regularizer=regularizer)(mixed)
        mixed = BatchNormalization(momentum=momentum, axis=3, name='LD2_BN_m')(mixed, training=isTraining)
        mixed = Activation("sigmoid", name='LD2_A_S_m')(mixed)
        
        ## Layer DE 3
        mixed = Conv2DTranspose(128, (3,3), padding="same", kernel_initializer=initializer, name='LD3_CT_m', kernel_regularizer=regularizer)(mixed)
        mixed = BatchNormalization(momentum=momentum, axis=3, name='LD3_BN_m')(mixed, training=isTraining)
        mixed = Activation("sigmoid", name='LD3_A_S_m')(mixed)
        
        ## Layer DE 4
        mixed = Conv2DTranspose(self.r, (5,5), padding="same", kernel_initializer=initializer, name='LD4_CT_m', kernel_regularizer=regularizer)(mixed)
        mixed = BatchNormalization(momentum=momentum, axis=3, name='LD4_BN_m')(mixed, training=isTraining)
        mixed = Activation("sigmoid", name='mixed')(mixed)
        
        self.model = Model(inputs=[pure_input, mixed_input], outputs=[abundances_pure, abundances_mixed, mixed])

    def compileModel(self, lr=1e-3):
        self.model.compile(
            optimizer=Adam(learning_rate=lr),
            loss = {
                "abundances_pure": CategoricalCrossentropy(from_logits=False),
                # "abundances_pure": MeanSquaredError(),
                "mixed": MeanSquaredError(),
                "abundances_mixed": None
            },
            loss_weights = {
                "abundances_pure": 1.0,
                # "abundances_pure": MeanSquaredError(),
                "mixed": 50.0,
                "abundances_mixed": 0.0
            }
        )
        
    def predict(self, pure_input, mixed_input):
        return self.model.predict({"i_p": pure_input, "i_m": mixed_input})
        
    def trainModel(self, pure_input, abundances_pure, mixed_input, epochs=200, batch_size=100, validation_split=.4):
        return self.model.fit(
            {"i_p": pure_input, "i_m": mixed_input},
            {"abundances_pure": abundances_pure, "mixed": mixed_input},
            epochs=epochs,
            validation_split=validation_split,
            batch_size=batch_size
        )

    def predict(self, pure_input, mixed_input):
        return self.model.predict(
            {"i_p": pure_input, "i_m": mixed_input}
        )
        
    def printModel(self, label = 'model'):
        plot_model(self.model, to_file=f'{label}.png', show_layer_activations=True, show_shapes=True)
        
    def summary(self):
        self.model.summary()


# model_pure, model_mixed = initModel()
# compileModel(model_pure, model_mixed)


# model.summary()
# history_pure = trainModelPure(model_pure, pure, abundances_pure)
# history_mixed = trainModelMixed(model_mixed, mixed)
