import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import csv
import os
import requests
import io
import zipfile
import shutil
import re
#import numpy as np
import pandas as pd
import tensorflow as tf
import splitfolders

from IPython.display import display
from matplotlib import pyplot as plt
from glob import glob
from json import dumps, loads

from tensorflow import expand_dims
from tensorflow.keras import preprocessing, models

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

#from aussys_tools import aussys_rb_thres, aussys_rb_images, tolerance_analysis, ratio2priori

plt.style.use('fivethirtyeight')

import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
    tn, fp, fn, tp = confusion_matrix(expected, predicted).ravel()

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
    sea_image_exp = ratio * (mission_duration * captures_per_second)
    nosea_image_exp = (1 - ratio) * (mission_duration * captures_per_second)

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


def tolerance_analysis(predict_proba, expected, fpr_tolerance=None, fnr_tolerance=None):
    """
    Parameters:
    `predict_proba` float array(m): probability of belonging to target class
    `expected` boolean array(m): state of target class to label
    `fpr_tolerance` (or `fnr_tolerance`) float: tolerance on the amount of false
            positives (negatives) in the real-world. The amount is given in
            terms of proportions on the events observed, independently on the
            class.

    Return:
    `fpr, fnr, thr` array: expected number of observations in each category, and
            suggested threshold.
    """
    if fpr_tolerance is not None:
        assert fnr_tolerance is None
        for threshold in np.sort(np.unique(predict_proba)):
            FPR, FNR = threshold_analysis(predict_proba, expected, threshold)
            if FPR <= fpr_tolerance:
                return FPR, FNR, threshold
        return 0, 1, 1

    if fnr_tolerance is not None:
        assert fpr_tolerance is None
        for threshold in -np.sort(-np.unique(predict_proba)):
            FPR, FNR = threshold_analysis(predict_proba, expected, threshold)
            if FNR <= fnr_tolerance:
                return FPR, FNR, threshold
        return 1, 0, 0

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

### GENERAL TOOLS ###

def run_analisys_metrics(data, thresholds = [ .25, 0.5, 0.75 ]):
    reports = dict(model=list(), dataset=list(), fold=list(), threshold=list(), accuracy=list(),
                   f1_score=list(), precision_nil=list(), precision_pod=list(), recall_nil=list(), recall_pod=list())
    
    for fold in range(data.fold.min(), data.fold.max()+1):
        expected = np.array(data[data.fold == fold]['expected'])
        predicted_proba = np.array(data[data.fold == fold]['predicted'])
    
        for threshold in thresholds:
            reports['model'].append(data[data.fold == fold]['model'].sample().values[0])
            reports['dataset'].append(data[data.fold == fold]['dataset'].sample().values[0])
            reports['fold'].append(fold)
            reports['threshold'].append(threshold)
    
            predicted = (predicted_proba > threshold)
            cr = classification_report(expected, predicted, target_names=['nil', 'pod'], output_dict=True)
    
            reports['accuracy'].append(cr['accuracy'])
            reports['f1_score'].append(round(f1_score(expected, predicted), 1))
            reports['precision_nil'].append(cr['nil']['precision'])
            reports['precision_pod'].append(cr['pod']['precision'])
            reports['recall_nil'].append(cr['nil']['recall'])
            reports['recall_pod'].append(cr['pod']['recall'])
    
    report = pd.DataFrame(reports)
    return report

def run_analisys(data):
    reports = dict(model=list(), dataset=list(), fold=list(), goal=list(),
                   goal_type=list(), fpr=list(), fnr=list(), thr=list())

    for fold in range(data.fold.min(), data.fold.max()+1):
        expected = np.array(data[data.fold == fold]['expected'])
        predicted_proba = np.array(data[data.fold == fold]['predicted'])

        for goal in [0.1, 0.2, 0.3]:
            for goal_type in ["fpr", "fnr"]:
                reports['model'].append(data[data.fold == fold]['model'].sample().values[0])
                reports['dataset'].append(data[data.fold == fold]['dataset'].sample().values[0])
                reports['fold'].append(fold)
                reports['goal'].append(goal)
                reports['goal_type'].append(goal_type)

                if goal_type == 'fpr':
                    fpr, fnr, thr = tolerance_analysis(predicted_proba, expected, fpr_tolerance=goal)
                else:
                    fpr, fnr, thr = tolerance_analysis(predicted_proba, expected, fnr_tolerance=goal)
                reports['fpr'].append(fpr)
                reports['fnr'].append(fnr)
                reports['thr'].append(thr)

    report = pd.DataFrame(reports)
    return report

def plot_result_metrics(data, thresholds = [ .25, 0.5, 0.75 ]):
    if not os.path.isdir('metrics'):
        os.mkdir('metrics')

    model_name = data['model'].sample().values[0]
    scenery = data['dataset'].sample().values[0]
    
    metrics = ['accuracy', 'precision_nil', 'precision_pod', 'recall_nil', 'recall_pod', 'f1_score']

    fig, ax = plt.subplots()
    fig.set_size_inches(28, 7)
    fig.suptitle(f"MODEL: {model_name} | SCENERY: {scenery}", fontsize=20)

    for i in range(len(metrics)):
        data_box = []
        for threshold in thresholds:
            data_metric = data[(data['threshold'] == threshold)]
            data_box.append(data_metric[metrics[i]])

        plt.subplot(1, 6, i + 1)
        plt.title(metrics[i])
        plt.boxplot(data_box, labels=thresholds, showfliers=False)
        plt.ylim(0.0, 1.1)

    plt.savefig(f"metrics/{model_name}_{scenery}_BP.png", format='png', bbox_inches='tight')

def compare_results(data, metrics, thresholds = [ .25, 0.5, 0.75 ]):
    if not os.path.isdir('tradeoffs'):
        os.mkdir('tradeoffs')

    model_name = data['model'].sample().values[0]
    scenery = data['dataset'].sample().values[0]
    
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 5)
    fig.suptitle(f"MODEL: {model_name} | SCENERY: {scenery}", fontsize=12)

    for i in range(len(metrics)):
        data_box = []
        for threshold in thresholds:
            data_metric = data[(data['threshold'] == threshold)]
            data_box.append(data_metric[metrics[i]])

        plt.subplot(1, len(metrics), i + 1)
        plt.title(metrics[i], fontsize=12)
        plt.boxplot(data_box, labels=thresholds, showfliers=False)
        plt.ylim(0.0, 1.1)

    plt.savefig(f"tradeoffs/{model_name}_{scenery}.png", format='png', bbox_inches='tight')

def false_rate(expected, predicted_proba, limiar):
    predicted = (predicted_proba > limiar)
    TP = np.sum(predicted & expected)
    TN = np.sum(~predicted & ~expected)
    FP = np.sum(predicted & ~expected)
    FN = np.sum(~predicted & expected)
    fnr = FN / (FN + TP)
    fpr = FP / (FP + TN)
    return fnr, fpr

def ROC_DET_val(data):
    list_thr = np.arange(0, 1, 0.005).tolist()

    roc_det = dict(model=list(), dataset=list(), fpr=list(), fnr=list())

    for fold in range(data.fold.min(), data.fold.max()+1):
        false_neg, false_pos = [],[]
        expected = np.array(data[data.fold == fold]['expected'])
        predicted_proba = np.array(data[data.fold == fold]['predicted'])

        for limiar in list_thr:
            fnr, fpr = false_rate(expected, predicted_proba, limiar)
            false_neg.append(fnr)
            false_pos.append(fpr)

        roc_det['model'].append(data['model'].sample().values[0])
        roc_det['dataset'].append(data['dataset'].sample().values[0])
        roc_det['fpr'].append(false_pos)
        roc_det['fnr'].append(false_neg)

    roc_det = pd.DataFrame(roc_det)
    return roc_det

def mean_std(data):
  results = dict(fpr_mean=list(), fpr_std=list(), fnr_mean=list(), fnr_std=list())
  for i in range(len(data['fpr'][0])):
    fnr, fpr = [],[]
    for j in range(len(data)):
      fnr.append(data['fnr'][j][i])
      fpr.append(data['fpr'][j][i])

    results['fpr_mean'].append(np.mean(fpr))
    results['fpr_std'].append(np.std(fpr))
    results['fnr_mean'].append(np.mean(fnr))
    results['fnr_std'].append(np.std(fnr))

  results = pd.DataFrame(results)
  return results

def plot_false_rates(df, max = 1):
    list_thr = np.arange(0, 1, 0.005).tolist()

    model_name = df['model'].sample().values[0]
    scenery = df['dataset'].sample().values[0]

    data = mean_std(df)
    index = []
    for i in range(len(list_thr)):
        if data['fpr_mean'][i] <= max and data['fnr_mean'][i] <= max:
            index.append(i)

    x0     = [list_thr[x] for x in index]
    Y1     = pd.Series([data['fpr_mean'][x] for x in index])
    Y1_std = pd.Series([data['fpr_std'][x] for x in index])
    Y2     = pd.Series([data['fnr_mean'][x] for x in index])
    Y2_std = pd.Series([data['fnr_std'][x] for x in index])

    plt.figure(figsize=(8,4))
    plt.rc('font', size=10)
    plt.title(f"MODEL: {model_name} | SCENERY: {scenery}", fontsize=12)
    plt.plot(x0, Y1, 'r-', linewidth='1', label = 'False positive rate')
    plt.fill_between(x0, Y1 - Y1_std, Y1 + Y1_std, color='r', alpha=0.2)
    plt.plot(x0, Y2, 'b-', linewidth='1', label = 'False negative rate')
    plt.fill_between(x0, Y2 - Y2_std, Y2 + Y2_std, color='b', alpha=0.2)
    plt.xlabel("Threshold", fontsize=12)
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.show

    plt.savefig(f"metrics/{model_name}_{scenery}_FR.png", format='png', bbox_inches='tight')

def predict(model, dataset, image_size):
    nil = glob(f'{dataset}/test/NIL/*.jpg')
    pod = glob(f'{dataset}/test/POD/*.jpg')

    X_test = nil + pod
    y_test = [ 0 for _ in range(len(nil)) ] + [ 1 for _ in range(len(pod)) ]
    y_pred = []
    y_pred_proba = []

    # predict dataset test
    id = []
    for i in range(len(X_test)):
        path_image = X_test[i]
        id.append(path_image.split('/')[-1])

        if len(image_size) == 2: image = preprocessing.image.load_img(path_image, target_size=image_size, color_mode="grayscale")
        else: image = preprocessing.image.load_img(path_image, target_size=image_size)
        image = expand_dims(image, 0)

        prediction = model.predict(image, verbose=0)
        y_pred_proba.append(prediction[:, 1][0])
        y_pred.append(prediction.argmax(axis=1)[0])

    return id, y_test, y_pred, y_pred_proba

def predict_model(model: str, dataset: str):
    """
    Generate the results for a single model in a specific dataset.
    Saves the result in a CSV file.
    """

    model_path = model.split('.zip')[0]
    with zipfile.ZipFile(model, 'r') as zip_ref:
        zip_ref.extractall(model_path)
    f = open(model_path+'/metadata.json')
    data = json.load(f)
    f.close()

    model_name   = model_path.split('/')[-1]
    dataset_name = dataset.split('/')[-1]

    model = models.load_model(model_path+'/model', custom_objects={'fbeta': fbeta})

    results = pd.DataFrame()
    for i in range(10):
        res_aux = pd.DataFrame()
        path = f'{dataset}/split_{i+1:02}'
        id, y_test, y_pred, y_pred_proba = predict(model, path, image_size = eval(data['model_input']))

        res_aux = pd.DataFrame({'id': id,
                                'model': model_name,
                                'dataset': dataset_name,
                                'fold': i+1,
                                'expected': y_test,
                                'predicted': y_pred_proba})

        results = pd.concat([results, res_aux], ignore_index=True)

    results.to_csv(f'results/results_{model_name}&{dataset_name}.csv', index = False)
    shutil.rmtree(model_path)

### API REQUESTS ###

def get_status(base_url = 'http://127.0.0.1:8000'):
    s = requests.get(f'{base_url}/state')
    state = s.json()
    print('MODELS:', state['MODELS'])
    print('SCENARIOS:', state['SCENARIOS'])
    print('RESULTS:', state['RESULTS'])

def upload_models(path, base_url = 'http://127.0.0.1:8000'):
    file = {'file': open(path, 'rb')}    
    response = requests.post(f'{base_url}/upload_model', files=file)
    message = response.json()
    print(message['msg'])

def upload_scenarios(path, base_url = 'http://127.0.0.1:8000'):
    file = {'file': open(path, 'rb')}
    response = requests.post(f'{base_url}/upload_scenarios', files=file)
    message = response.json()
    print(message['msg'])

def run_predictions(model, scenery, base_url = 'http://127.0.0.1:8000'):
    params = {'scenery': scenery, 'model': model}
    response = requests.post(f'{base_url}/run_predictions', params=params)
    message = response.json()
    print(message['msg'])
    
def download_results(model, scenery, base_url = 'http://127.0.0.1:8000'):
    params = {'scenery': scenery, 'model': model}
    response = requests.post(f'{base_url}/download_results', params=params)
    response_decoded = response.content.decode('utf-8')
    csv_file = csv.reader(io.StringIO(response_decoded))
    df = pd.DataFrame(csv_file)

    if len(df) == 1:
        return print(df[0][0])
        
    else:
        df = df.rename(columns=df.iloc[0]).drop(df.index[0])
        df = df.astype({'fold':'int','expected':'int','predicted':'float'})
        return df

### REPORT ###

def full_report(data):
    report_metrics = run_analisys_metrics(data)
    plot_result_metrics(report_metrics)

    ######################

    report = run_analisys(data)
    report.rename(columns={"goal": "Tolerância", "goal_type": "Métrica", "fpr": "FPR", "fnr": "FNR", "thr": "Threshold"}, inplace=True)
    report['Métrica'].replace('fpr', 'FPR', inplace=True, regex=True)
    report['Métrica'].replace('fnr', 'FNR', inplace=True, regex=True)
    
    report_fnr = report.drop(report[report['Métrica'] == 'FPR'].index)
    report_fpr = report.drop(report[report['Métrica'] == 'FNR'].index)
    report_fnr = report_fnr.groupby(['Tolerância', 'Métrica'])[['FPR', 'Threshold']].describe().loc[:,(slice(None),['mean', 'std'])].apply(lambda s: s.apply('{0:.2f}'.format))
    report_fpr = report_fpr.groupby(['Tolerância', 'Métrica'])[['FNR', 'Threshold']].describe().loc[:,(slice(None),['mean', 'std'])].apply(lambda s: s.apply('{0:.2f}'.format))
    
    display(report_fnr)
    print()
    display(report_fpr)
    print()

    ######################

    rates = ROC_DET_val(data)
    plot_false_rates(rates)
