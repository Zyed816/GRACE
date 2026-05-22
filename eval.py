import os

import numpy as np

from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize, OneHotEncoder


def prob_to_one_hot(y_pred):
    ret = np.zeros(y_pred.shape, bool)
    indices = np.argmax(y_pred, axis=1)
    for i in range(y_pred.shape[0]):
        ret[i][indices[i]] = True
    return ret


def print_statistics(statistics, function_name):
    print(f'(E) | {function_name}:', end=' ')
    for i, key in enumerate(statistics.keys()):
        mean = statistics[key]['mean']
        std = statistics[key]['std']
        print(f'{key}={mean:.4f}+-{std:.4f}', end='')
        if i != len(statistics.keys()) - 1:
            print(',', end=' ')
        else:
            print()


def _label_classification_once(embeddings, y, ratio, random_state=None):
    X = embeddings.detach().cpu().numpy()
    Y = y.detach().cpu().numpy()
    Y = Y.reshape(-1, 1)
    onehot_encoder = OneHotEncoder(categories='auto').fit(Y)
    Y = onehot_encoder.transform(Y).toarray().astype(bool)

    X = normalize(X, norm='l2')

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        Y,
        test_size=1 - ratio,
        random_state=random_state,
    )

    logreg = LogisticRegression(solver='liblinear')
    c = 2.0 ** np.arange(-10, 10)
    eval_n_jobs = int(os.environ.get('GRACE_EVAL_N_JOBS', '1'))

    clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
                       param_grid=dict(estimator__C=c), n_jobs=eval_n_jobs, cv=5,
                       verbose=0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)
    y_pred = prob_to_one_hot(y_pred)

    micro = f1_score(y_test, y_pred, average="micro")
    macro = f1_score(y_test, y_pred, average="macro")

    return {
        'F1Mi': micro,
        'F1Ma': macro
    }


def label_classification(embeddings, y, ratio, eval_repeats=3, eval_seed=None):
    eval_repeats = max(1, int(eval_repeats))
    results = []
    for idx in range(eval_repeats):
        random_state = None if eval_seed is None else int(eval_seed) + idx
        results.append(_label_classification_once(embeddings, y, ratio, random_state=random_state))

    statistics = {}
    for key in results[0].keys():
        values = [r[key] for r in results]
        statistics[key] = {
            'mean': np.mean(values),
            'std': np.std(values)}
    print_statistics(statistics, 'label_classification')
    return statistics
