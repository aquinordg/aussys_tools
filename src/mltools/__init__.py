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
    `FPR(fall-out), FNR(miss rate)`: array
    
    """
    
    predicted = (predict_proba > threshold)
    cm = confusion_matrix(expected, predicted)
    
    FPR = 1 - (cm[0, 0]/sum(cm[0, :]))
    FNR = 1 - (cm[1, 1]/sum(cm[1, :]))

    return FPR, FNR
    
def aussys_rb_thres(predict_proba, expected, threshold, mission_duration, captures_per_second, sea_nosea_ratio, print_mode=True):
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
    `sea_fpr, nosea_fnr`: array
    
    """
    ratio = sea_nosea_ratio/(sea_nosea_ratio + 1)
    sea_image_exp = (1 - ratio) * (mission_duration * captures_per_second)
    nosea_image_exp =  ratio * (mission_duration * captures_per_second)

    
    FPR, FNR = threshold_analysis(predict_proba, expected, threshold)
    
    sea_fpr = int(sea_image_exp * FPR)
    nosea_fnr = int(nosea_image_exp * FNR)
    
    if print_mode == True:
        print('>>> RELATÓRIO:')
        print(f'- Espera-se que {sea_fpr} imagens `no sea` sejam identificadas de forma equivocada.')
        print(f'- Estima-se que {nosea_fnr} imagens `no sea` deverão passar despercebidas.')
        
        return
    
    else:
    
        return sea_fpr, nosea_fnr

def get_rates_b_thres(predict_proba, expected, rate_type, ref, sen):

    for threshold in np.arange(0, 1, 10**(-sen)):
        predicted = (predict_proba > threshold)
        cm = confusion_matrix(expected, predicted)
        FPR_t = round(1 - (cm[0, 0]/sum(cm[0, :])), sen)
        FNR_t = round(1 - (cm[1, 1]/sum(cm[1, :])), sen)
        
        if (rate_type == "FPR") and (FPR_t <= ref):
            break
            
        if (rate_type == "FNR") and (FNR_t >= ref):
            break
    
    return FPR_t, FNR_t, threshold

def aussys_rb_images(predict_proba, expected, mission_duration, captures_per_second, sea_nosea_ratio, sen, sea_fpr=None, nosea_fnr=None, print_mode=True):
    
    ratio = sea_nosea_ratio/(sea_nosea_ratio + 1)
    sea_image_exp = (1 - ratio) * (mission_duration * captures_per_second)
    nosea_image_exp = ratio * (mission_duration * captures_per_second)
        
    if sea_fpr is not None:
        FPR = round(sea_fpr/sea_image_exp, sen)
        FPR_t, FNR_t, threshold = get_rates_b_thres(predict_proba, expected, rate_type='FPR', ref=FPR, sen=sen)
        nosea_fnr_t = int(nosea_image_exp * FNR_t)
        sea_fpr_res = [nosea_fnr_t, threshold]
        
    if nosea_fnr is not None:
        FNR = round(nosea_fnr/nosea_image_exp, sen)
        FPR_t, FNR_t, threshold = get_rates_b_thres(predict_proba, expected, rate_type='FNR', ref=FNR, sen=sen)
        sea_fpr_t = int(sea_image_exp * FPR_t)
        nosea_fnr_res = [sea_fpr_t, threshold]
        
    if (print_mode == True) and (sea_fpr is not None):
        print('>>> RELATÓRIO SEA FPR:')
        print(f'- Dado que podem ser aceitas até {sea_fpr} imagens `no sea` identificadas de forma equivocadas,') 
        print(f'convêm-se utilizar um threshold de {sea_fpr_res[1]},')
        print(f'e espera-se que {sea_fpr_res[0]} imagens passem despercebidas.\n')
        
    if (print_mode == True) and (nosea_fnr is not None):
        print('>>> RELATÓRIO NOSEA FNR:')
        print(f'- Dado que podem ser aceitas até {nosea_fnr} imagens `no sea` passarem despercebidas,')
        print(f'convêm-se utilizar um threshold de {nosea_fnr_res[1]},')
        print(f'e espera-se que {nosea_fnr_res[0]} imagens sejam identificadas de forma equivocadas.\n')
        
    if (sea_fpr is not None) and (nosea_fnr is not None) and (print_mode==False):
        return np.array([sea_fpr_res, nosea_fnr_res])   

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