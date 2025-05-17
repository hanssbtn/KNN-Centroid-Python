import numpy as np
import pandas as pd
from numpy import linalg as la
from typing import *
import functools

class KNN:
	def __init__(self, k: int = 3) -> Self:
		assert k % 2 == 1 and k > 0, "Invalid k value"
		self.__k = k

	def fit(self, priors: pd.DataFrame) -> None:
		self.attributes = [col for col in priors.columns[:-1]]	
		# Assume last column represents the class
		classes = priors.columns[-1]
		
		# Typecast all rows except the class rows as floats
		for col in priors.columns[:-1]:
			priors[col] = priors[col].astype(np.float64)
		
		def normalize(df: pd.DataFrame):
			df.iloc[:, :-1] = df.iloc[:, :-1] / np.linalg.norm(df.iloc[:, :-1], axis=1, keepdims=True)
			return df

		# Group by classes, then normalize, reset index to convert back to dataframe, and use the class column as index
		res: pd.DataFrame = priors \
			.groupby(classes)[priors.columns[:]] \
			.apply(normalize) \
			.reset_index(drop=True)
		self.priors = res[res.columns[:-1]]
		self.categories = res[res.columns[-1]].to_numpy(np.int64)
		grouped = res.groupby(classes)
		# Create a new dataframe for the centroid
		centroids: pd.DataFrame = grouped[res.columns[:]] \
			.apply(functools.partial(np.mean, axis=0)) \
		
		# Store only the arrays
		self.centroids_ = centroids[centroids.columns[:-1]].to_numpy()
		# Convert the category to an integer datatype
		self.classes = centroids[centroids.columns[-1]].to_numpy(np.int64)

		# Calculate inverse covariance matrix for mahalanobis distance
		try:
			self.inv_cov: np.ndarray[np.ndarray[np.number]] = np.linalg.inv(self.priors.cov())
		except np.linalg.LinAlgError:
			cov = self.priors.cov()
			self.inv_cov = np.linalg.inv(cov + np.identity(cov.shape[0]) * 1e-11)

	def predict(self, test_data: pd.DataFrame | Iterable, metric: str = 'euclidian') -> np.ndarray[np.number] | np.number:
		if isinstance(test_data, np.ndarray):
			td = test_data.astype(np.float64)
		elif isinstance(test_data, pd.DataFrame):
			td = test_data.to_numpy(dtype=np.int64)
		elif isinstance(test_data, Iterable):
			# Assume data is 1-dimensional
			td = np.array(test_data, dtype=np.float64)
		else:
			raise ValueError(f"test_data is not an Iterable / DataFrame (got {type(test_data)})")
		dims = len(td.shape)
		assert (dims == 2 and td.shape[1] == self.priors.shape[1]) or (dims == 1 and td.shape[0] == self.priors.shape[1])
		match dims:
			case 2:
				td = td.astype(np.float64) / np.linalg.norm(td, axis=1)[:, np.newaxis]
			case 1:
				td = (td/np.linalg.norm(td))[np.newaxis, :]
			case _:
				raise IndexError(f"Cannot convert td to 2D matrix (got {dims})")
		ret: np.ndarray[np.number]
		diff = td[:, np.newaxis, :] - self.priors.to_numpy()[np.newaxis, :, :]
		match (metric):
			case 'euclidian' | 'e':
				dist = np.linalg.norm(diff, axis=2)
			case 'mahalanobis' | 'm':
				dist = np.sqrt(np.einsum('ijk,kl,ijl->ij', diff, self.inv_cov, diff))
			case 'centroid' | 'c':
				diff = td[:, np.newaxis, :] - self.centroids_[np.newaxis, :, :]
				dist = np.linalg.norm(diff, axis=2)

				nn_indices = np.argsort(dist, axis=1)[:, :1]
				# Equal to
				# neighbors = np.take_along_axis(self.categories[np.newaxis, :], nn_indices, axis=1)
				neighbors = self.classes[nn_indices.flatten()].reshape(nn_indices.shape)
				ret = np.apply_along_axis(KNN.__classify, 1, neighbors)
				if ret.shape[0] == 1:
					return ret[0]
				else:
					return ret
			case 'cosine' | 'cs':
				similarities = np.dot(td, self.priors.T)
				# Equal to
				# neighbors = np.take_along_axis(self.categories[np.newaxis, :], max_indices, axis=1)
				max_indices = np.argsort(similarities, axis=1)[:, -self.__k:]
				neighbors = self.categories[max_indices.flatten()].reshape(max_indices.shape)
				ret = np.apply_along_axis(KNN.__classify, 1, neighbors)
				if ret.shape[0] == 1:
					return ret[0]
				else:
					return ret
			case _:
				raise ValueError(f"Invalid metric (got {metric})")
		nn_indices = np.argsort(dist, axis=1)[:, :self.__k]

		# Equal to
		# neighbors = np.take_along_axis(self.categories[np.newaxis, :], nn_indices, axis=1)
		neighbors = self.categories[nn_indices.flatten()].reshape(nn_indices.shape)
		ret = np.apply_along_axis(KNN.__classify, 1, neighbors)
		if ret.shape[0] == 1:
			return ret[0]
		else:
			return ret
	
	@staticmethod
	def __classify(neighbors: np.ndarray[np.int64]):
		cls, cnt = np.unique_counts(neighbors)
		return cls[np.argmax(cnt)]

	def __euclidian_nearest_neighbor(self, x: pd.Series) -> np.int64:
		diff = self.priors - np.repeat([x.to_numpy()], self.priors.shape[0], axis=0)
		dist = np.linalg.norm(diff.to_numpy(), axis=1)
		min_indices = np.argsort(dist)[:self.__k]
		return KNN.__classify(self.categories[min_indices])
	
	def __mahalanobis_nearest_neighbor(self, x: pd.Series) -> np.int64:
		diff = self.priors - np.repeat([x.to_numpy()], self.priors.shape[0], axis=0)
		dist = diff.apply(lambda r: np.sqrt(np.dot(np.dot(r, self.inv_cov), r)), axis=1)
		min_indices = np.argsort(dist)[:self.__k]
		return KNN.__classify(self.categories[min_indices])
