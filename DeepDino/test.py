from keras.models import load_model
import numpy as np

def test(modelpath,env):
    model = load_model(modelpath)
    obs, _, done, _ = env.step(0)
    while not done:
        Q = model.predict(obs[np.newaxis,:,:,:])
        action = np.argmax(Q)
        obs, _, done, _ = env.step(action)