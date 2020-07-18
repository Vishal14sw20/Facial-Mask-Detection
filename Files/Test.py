import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

def evalute_report(model,testX,testY,hyper_param):
    print("[INFO] evaluating network...")
    predIdxs = model.predict(testX, batch_size=hyper_param['batch_size'])
    # for each image in the testing set we need to find the index of the
    # label with corresponding largest predicted probability
    #print(predIdxs)
    probs = predIdxs[:,1]
    precision, recall, _ = precision_recall_curve(testY,probs)
    predIdxs = np.argmax(predIdxs, axis=1)
    fig,axs = plt.subplots(2)
    axs[0].plot(recall,precision,marker='.')
    axs[0].xlabel = 'recall'
    axs[0].ylabel = 'precision'
    axs[0].legend('Precision-Recall curve')

    fpr, tpr, _ = roc_curve(testY, probs)
    axs[1].plot(fpr,tpr,marker='.')
    axs[1].xlabel = 'False Positive Rate'
    axs[1].ylabel = 'True Positive Rate'
    axs[1].legend('ROC Curve')

    # show a nicely formatted classification report
    print(classification_report(testY.argmax(axis=1), predIdxs))
    plt.show()