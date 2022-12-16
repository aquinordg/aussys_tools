import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def threshold_analysis(X, y, tss, model, idc, thr, random_state):
    """
    Parameters:
    `X` array(n, m): approximate number of examples
    `y` array(n): number of clusters
    `tss` float[0,1]: train test split
    `model` model: sklearn model
    `idc` int: index of class to be analyzed
    `thr` float[0,1]: threshold
    `random_state` int: random seed
    
    Return:
    `FPR, FNR`: array
    
    """
    
    assert idc in np.unique(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tss, random_state=random_state)
    model.fit(X_train, y_train)
    predicted = (model.predict_proba(X_test)[:, idc] > thr)
    expected = y_test == idc
    
    cm = confusion_matrix(expected, predicted)

    return [1 - (cm[j, j]/sum(cm[j, :])) for j in range(len(cm))]