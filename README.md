# Package mltools

Python package [mltools](https://github.com/aquinordg/mltools): Machine Learning Tools.

<div style="text-align: justify"> This tool assists in the analysis and decision making in binary classification scenarios, mainly in applications for detecting drifting objects at sea. </div>

## Functions

### Probability threshold based report

<div style="text-align: justify"> This function informs the amount of 'no sea' images identified in a mistaken and unnoticed way, given a certain threshold of probability established as a parameter. For this, the function uses a membership probability matrix and expected values, given a given class, in addition to some mission information. The results can be shown by a report printed on screen or acquired directly. </div>

```markdown

aussys_rb_thres(predict_proba,
                expected,
                mission_duration,
                captures_per_second,
                n_sea_exp,
                threshold,
                print_mode=True)

```

#### Parameters

* **predict_proba**: _array-like of float in range [0.0, 1.0] and shape m_<br/>
> Probabilities of the examples belonging to the target class.

* **expected**: _array-like of bool and shape m_<br/>
> Expected classes of each of the examples.

* **mission_duration**: _int_<br/>
> Mission duration in seconds.

* **captures_per_second**: _int_<br/>
> Number of captures per second.

* **n_sea_exp**: _int_<br/>
> Expected number of 'sea' images for 1 'nosea'. <br/>
> **Ex.:** 1('nosea') : _n_sea_exp_.

* **threshold**: _float in range [0.0, 1.0]_<br/>
> User-defined membership probability threshold.

* **print_mode**: _bool_<br/>
> Screen report (**True**) or values (**False**).

#### _Returns_

* **sea_fpr**: _int_<br/>
> Number of wrongly identified 'no sea' images.

* **nosea_fnr**: _int_<br/>
> Number of 'no sea' images that should go unnoticed.

### Report based on the number of images

<div style="text-align: justify"> This function informs updated amounts of 'nosea' images misidentified or passed unnoticed considered acceptable and the adequate threshold, according to the chosen input. As with the previous function, the membership probability matrix and expected values are used, given a given class, in addition to some mission information. The method finds the new threshold through a greedy search in all possible scenarios based on a sensitivity parameter. The results can be shown by a report printed on screen or acquired directly. </div>

```markdown

aussys_rb_images(predict_proba,
                 expected,
                 mission_duration,
                 captures_per_second,
                 n_sea_exp,
                 sea_fpr=None,
                 nosea_fnr=None,
                 print_mode=True)

```
#### Parameters

* **predict_proba**: _array-like of float in range [0.0, 1.0] and shape m_<br/>
> Probabilities of examples belonging to the target class.

* **expected**: _array-like of bool and shape m_<br/>
> Expected classes of each of the examples.

* **mission_duration**: _int_<br/>
> Mission duration in seconds.

* **captures_per_second**: _int_<br/>
> Number of captures per second.

* **n_sea_exp**: _int_<br/>
> Expected number of 'sea' images for 1 'nosea'. <br/>
> **Ex.:** 1('nosea') : _n_sea_exp_.

* **sea_fpr**: _int_<br/>
> Number of 'no sea' images where it is acceptable to be misidentified.

* **nosea_fnr**: _int_<br/>
> Number of 'no sea' images where it is acceptable to go unnoticed.

* **print_mode**: _bool_<br/>
> Screen report (**True**) or values (**False**).

#### _Retornos_

* **new_sea_fpr, threshold**: _array-like_<br/>
> Number of wrongly identified 'no sea' images, given the indicated 'nosea_fnr' value and the corresponding appropriate threshold value.

* **new_nosea_fnr, threshold**: _array-like_<br/>
> Number of 'no sea' images that should go unnoticed, given the value of 'sea_fpr' indicated and the appropriate threshold.
