import numpy as np
from sklearn.metrics import classification_report

def evalute_report(model,testX,testY,hyper_param):
    print("[INFO] evaluating network...")
    predIdxs = model.predict(testX, batch_size=hyper_param['batch_size'])
    # for each image in the testing set we need to find the index of the
    # label with corresponding largest predicted probability
    predIdxs = np.argmax(predIdxs, axis=1)
    # show a nicely formatted classification report
    print(classification_report(testY.argmax(axis=1), predIdxs))