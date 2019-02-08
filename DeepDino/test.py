from keras.models import load_model
import numpy as np

def test(modelpath,env):
    model = load_model(modelpath)
    model.summary()
    high_score = 0
    for i in range(10):
        env.reset()
        obs, _, done, _ = env.step(0)
        while not done:
                Q = model.predict(obs[np.newaxis,:,:,:])
                action = np.argmax(Q)
                obs, _, done, _ = env.step(action)

        score = env.game.playerDino.score

        high_score = score if score > high_score else high_score
        print(high_score)