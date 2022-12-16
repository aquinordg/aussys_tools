import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def threshold_analysis(predict_proba, expected, threshold):
    """
    Parameters:
    `predict_proba` float array(m): probability of belonging to target class
    `expected` boolean array(m): state of target class to label
    `threshold` float[0,1]: threshold
    
    Return:
    `FPR, FNR`: array
    
    """
    
    predicted = (predict_proba > threshold)
    cm = confusion_matrix(expected, predicted)

    return [1 - (cm[j, j]/sum(cm[j, :])) for j in range(len(cm))]
    
def aussys_thres_report(predict_proba, expected, threshold, mission_duration, captures_per_second, sea_nosea_ratio, print_mode = True):
    """
    Parameters:
    `predict_proba` float array(m): probability of belonging to target class
    `expected` boolean array(m): state of target class to label
    `threshold` float[0,1]: threshold
    
    `mission_duration` model: duration of mission
    `captures_per_second` int: number of captures per second
    `sea_nosea_ratio` float[0,1]: sea/nosea ratio
    `print_mode` int: report or just values
    
    Return:
    `FPR, FNR`: array
    
    """
    
    sea_image_exp = (1 - sea_nosea_ratio/(sea_nosea_ratio + 1)) * (mission_duration * captures_per_second)
    nosea_image_exp = (sea_nosea_ratio/(sea_nosea_ratio + 1)) * (mission_duration * captures_per_second)

    
    ta = threshold_analysis(predict_proba, expected, threshold)
    
    sea_fpr = int(sea_image_exp * ta[0])
    nosea_fnr = int(nosea_image_exp * ta[1])
    
    if print_mode == True:
        print('>>> REPORT:')
        print(f'- Espera-se que {sea_fpr} imagens `no sea` sejam identificadas de forma equivocada.')
        print(f'- Estima-se que {nosea_fnr} imagens `no sea` deverão passar despercebidas.')
        
        return
    
    else:
    
        return sea_fpr, nosea_fnr
        
def threshold_analysis_gen(X, y, tss, model, idc, thr, random_state):
    """
    Parameters:
    `X` array(m, n): data
    `y` array(m): labels
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