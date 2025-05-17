import numpy as np
import pandas as pd
from numpy import linalg as la
from typing import *
import functools
from model import KNN

def split_data(df: pd.DataFrame, folds: int | np.int64) -> Generator[tuple[pd.DataFrame, pd.DataFrame]]:
	row_count = df.shape[0]
	fold_size = row_count // folds
	indices = np.arange(row_count)
	splits = [(indices[k * fold_size:(k + 1) * fold_size], np.concatenate((indices[:k * fold_size], indices[(k + 1) * fold_size:]))) for k in range(folds)]
	for train, test in splits:	
		yield df.loc[train], df.loc[test]

def calculate_macro_metrics(prediction: np.ndarray[np.number], actual: np.ndarray[np.number]) -> tuple[np.float64, np.float64, np.float64]:
	assert len(actual.shape) == 1 and actual.shape == prediction.shape, f"{actual.shape} {prediction.shape}"
	classes, counts = np.unique_counts(actual)
	positives = [ np.where(prediction == cls)[0] for cls in classes ]
	true_positives = [ np.sum(prediction[p] == actual[p]) for p in positives ]
	accuracy = np.sum(true_positives)/len(actual)
	recall = [ tp/len(p) if len(p) > 0 else 0 for p, tp in zip(positives, true_positives) ]
	precision = [ tp/count if count > 0 else 0 for tp, count in zip(true_positives, counts) ]
	assert np.sum(np.isnan(precision)) == 0 and np.sum(np.isnan(recall)) == 0, f"precision: {precision}, recall: {recall}"
	return accuracy, np.mean(precision), np.mean(recall)

if __name__ == "__main__":
	from timeit import timeit

	setup = """
import numpy as np
import pandas as pd
from numpy import linalg as la

rng = np.random.Generator(np.random.PCG64DXSM())
gc.collect()
gc.collect()
gc.disable()

def calculate_macro_metrics(prediction: np.ndarray[np.number], actual: np.ndarray[np.number]) -> np.float64:
	assert len(actual.shape) == 1 and actual.shape == prediction.shape
	classes, counts = np.unique_counts(actual)
	# Assume classes are sorted
	positives = [np.where(prediction == cls)[0] for cls in classes]
	true_positives = [np.sum(prediction[p] == actual[p]) for p in positives]
	accuracy = np.sum(true_positives)/len(actual)
	precision = [tp/len(p) for p, tp in zip(positives, true_positives)]
	recall = [tp/count for tp, count in zip(true_positives, counts)]
	assert np.sum(np.isnan(precision)) == 0 and np.sum(np.isnan(recall)) == 0
	return accuracy, np.mean(precision), np.mean(recall)
"""

	setup2 = """
import numpy as np
import pandas as pd
from numpy import linalg as la

rng = np.random.Generator(np.random.PCG64DXSM())
from sklearn.metrics import accuracy_score, precision_score, recall_score
gc.collect()
gc.collect()
gc.disable()
"""

	code = """
pred = rng.integers(0, 3, 1000)
actual = pred.copy()
indices = rng.choice(np.arange(pred.shape[0]), 100, False)
actual[indices] = (actual[indices] + 1) % 3
res = calculate_macro_metrics(pred, actual)
"""

	code2 = """
pred = rng.integers(0, 3, 1000)
actual = pred.copy()
indices = rng.choice(np.arange(pred.shape[0]), 100, False)
actual[indices] = (actual[indices] + 1) % 3

acc, prec, rec = accuracy_score(actual, pred),  precision_score(actual, pred, average='macro'), recall_score(actual, pred, average='macro')
"""

	import gc

	gc.enable()
	gc.collect()
	gc.collect()
	gc.disable()
	t1, t2  = timeit(code, setup, number=100000), timeit(code2, setup2, number=100000)
	print(t1, t2, f"t1/t2 = {t1/t2}")