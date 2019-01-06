import keras
from keras.models import Model
from keras.layers import Conv2D, Dense,Flatten, Input, Lambda
from keras.optimizers import Adam
import keras.backend as K

### MODEL HYPERPARAMETERS
learning_rate =  0.0002      # Alpha (aka learning rate)

"""
The action that this layer takes is a one hot encoded vector. It multiplies the Q vector with this action and sums over
the correct axis to "select" the Q value corresponding to the action.
"""
def Q_layer(merged):
    action_input = merged[0]
    y = merged[1]
    return K.expand_dims(K.sum(y * action_input, axis=1))

"""
Returns two models : one that takes a stacked state and returns a vector of Q values corresponding to
each action and a model that takes a stacked state and an action and returns the corresonding Q value.
The first one is for selecting the action that maximises the Q value and the second one is used for training
the model.
"""

def get_models():

    observation_input = Input(shape=(100,100,4), name="4_stacked_states")
    y = Conv2D(16, (8,8), strides=(4,4), activation="relu")(observation_input)
    y = Conv2D(32, (8,8), strides=(2,2), activation="relu")(y)
    y = Flatten()(y)
    y = Dense(256, activation="relu")(y)
    y = Dense(3)(y)

    base_model = Model(observation_input,y)

    action_input = Input(shape=(3,), name="action")
    
    Q = Lambda(Q_layer, name="Q_value")([action_input, y])

    Q_model = Model(inputs=[observation_input, action_input], outputs=Q)

    Q_model.compile(optimizer=Adam(learning_rate), loss="mse")

    return base_model, Q_model