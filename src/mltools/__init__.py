import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def threshold_analysis(predict_proba, expected, threshold):
    """

    Function used by another function (`aussys_rb_thres`)

    Parameters:
    `predict_proba` float array(m): probability of belonging to target class
    `expected` boolean array(m): state of target class to label
    `threshold` float[0,1]: threshold

    Return:
    `FPR`(fall-out) float[0,1]: False Positive Rate
    `FNR`(miss rate) float[0,1]: False Negative Rate

    """

    predicted = (predict_proba > threshold)
    tn, fp, fn, tp = confusion_matrix(expected, predicted)

    FPR = fp / (fp + tn)
    FNR = fn / (fn + tp)

    return FPR, FNR

def aussys_rb_thres(predict_proba, expected, mission_duration, captures_per_second, n_sea_exp, threshold, print_mode=True):
    """

    Aussys report by threshold

    This function informs the misidentified `no sea` and
    unidentified `no sea` images given a threshold. For this
    is required a probability array of belonging and the
    expected values of the class. Also, mission information and
    parameters are required. The results can be shown by printed report or
    just values.

    Parameters:
    `predict_proba` float array(m): probability of belonging to target class
    `expected` boolean array(m): state of target class to label

    `mission_duration` int: duration of mission in seconds
    `captures_per_second` int: number of captures per second
    `n_sea_exp` int: expected number of 'sea' images for 1 'nosea'.  Ex.: 1(nosea):n_sea_exp

    `threshold` float[0,1]: threshold

    `print_mode` int: report or values

    Return:
    `sea_fpr` int: misidentified `no sea` images
    `nosea_fnr` int: unidentified `no sea` images

    """
    ratio = n_sea_exp/(n_sea_exp + 1)
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

def get_rates_b_thres(predict_proba, expected, rate_type, ref):
    """

    Function used by another function (`aussys_rb_images`)

    Parameters:
    `predict_proba` float array(m): probability of belonging to target class
    `expected` boolean array(m): state of target class to label

    `rate_type` string: 'FPR' for False Positive Rate or 'FNR' for False Negative Rate
    `ref` model: FPR or FNR reference values to the search

    Return:
    `FPR_t` float: new value of FPR_t
    `FNR_t` float: new value of FNR_t
    `threshold` float: new value of threshold

    """
    for threshold in np.arange(0, 1, 10**(-2)):
        predicted = (predict_proba > threshold)
        cm = confusion_matrix(expected, predicted)
        FPR_t = round(1 - (cm[0, 0]/sum(cm[0, :])), 2)
        FNR_t = round(1 - (cm[1, 1]/sum(cm[1, :])), 2)

        if (rate_type == "FPR") and (FPR_t <= ref):
            break

        if (rate_type == "FNR") and (FNR_t >= ref):
            break

    return FPR_t, FNR_t, threshold


def ratio2priori(positive_count=1, negative_count=1):
    return positive_count / (positive_count + negative_count)


def tolerance_analysis(predict_proba, expected, positive_priori, fp_tolerance=None, fn_tolerance=None):
    """
    Parameters:
    `predict_proba` float array(m): probability of belonging to target class
    `expected` boolean array(m): state of target class to label
    `positive_priori` float: probability of finding a positive sample in real-world situation
    `fp_tolerance` (or `fn_tolerance`) float: tolerance on the amount of false
            positives (negatives) in the real-world. The amount is given in
            terms of proportions on the events observed, independently on the
            class.

    Return:
    `fp, fn, thr` array: expected number of observations in each category, and
            suggested threshold.
    """
    if fp_tolerance is not None:
        assert fn_tolerance is None
        for threshold in np.sort(np.unique(predict_proba)):

            tn, fp, fn, tp = confusion_matrix(expected, predict_proba > threshold)
            FPR = fp / (fp + tn)
            FNR = fn / (fn + tp)

            if FPR * (1 - positive_priori) <= fp_tolerance:
                return fp_tolerance, FNR * positive_priori, threshold

    if fn_tolerance is not None
        assert fp_tolerance is None
        for threshold in -np.sort(-np.unique(predict_proba)):

            tn, fp, fn, tp = confusion_matrix(expected, predict_proba > threshold)
            FPR = fp / (fp + tn)
            FNR = fn / (fn + tp)

            if FNR * positive_priori <= fn_tolerance:
                return FPR * (1 - positive_priori), fn_tolerance, threshold

def aussys_rb_images(predict_proba, expected, mission_duration, captures_per_second, n_sea_exp, sea_fpr=None, nosea_fnr=None, print_mode=True):
    """

    Aussys report by images

    This function informs new values for acceptable misidentified `no sea` or
    unidentified `no sea` images including a suitable threshold.
    For this is required a probability array of belonging and the
    expected values of the class. Also, mission information and
    parameters are required. The method find the new threshold
    by a greedy search in all possible scenarios using determined sensibility.
    The results can be shown by printed report or just values.

    Parameters:
    `predict_proba` float array(m): probability of belonging to target class
    `expected` boolean array(m): state of target class to label

    `mission_duration` int: duration of mission in seconds
    `captures_per_second` int: number of captures per second
    `n_sea_exp` int: expected number of 'sea' images for 1 'nosea'.  Ex.: 1(nosea):n_sea_exp

    `sea_fpr` int: misidentified `no sea` images (desired)
    `nosea_fnr` int: unidentified `no sea` images (desired)

    `print_mode` int: report or values

    Return:
    `sea_fpr_res, nosea_fnr_res` array: new values for sea_fpr and nosea_fnr, both with threshold

    """
    ratio = n_sea_exp/(n_sea_exp + 1)
    sea_image_exp = (1 - ratio) * (mission_duration * captures_per_second)
    nosea_image_exp = ratio * (mission_duration * captures_per_second)

    if sea_fpr is not None:
        FPR = round(sea_fpr/sea_image_exp, 2)
        FPR_t, FNR_t, threshold = get_rates_b_thres(predict_proba, expected, rate_type='FPR', ref=FPR)
        nosea_fnr_t = int(nosea_image_exp * FNR_t)
        sea_fpr_res = [nosea_fnr_t, round(threshold, 2)]

    if nosea_fnr is not None:
        FNR = round(nosea_fnr/nosea_image_exp, 2)
        FPR_t, FNR_t, threshold = get_rates_b_thres(predict_proba, expected, rate_type='FNR', ref=FNR)
        sea_fpr_t = int(sea_image_exp * FPR_t)
        nosea_fnr_res = [sea_fpr_t, round(threshold, 2)]

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
