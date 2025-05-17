import numpy as np
import pandas as pd
import functools
from typing import *
import math
from collections import deque
from heapq import *

class Node:
	def __init__(self, array: np.ndarray[np.number], category: np.int64) -> Self:
		self.array: np.ndarray[np.number] = array
		self.category: np.int64 = category 
	def __repr__(self) -> str:
		return f"({self.array}, {self.category})"

class NodePair:
	def __init__(self, node: Node, distance: np.number) -> Self:
		self.node = node
		self.distance = distance
	def __repr__(self) -> str:
		return f"{self.node}, distance: {self.distance:.2f}"
	def __lt__(self, other: Self) -> bool:
		return self.distance > other.distance
	def __iter__(self) -> Generator[np.ndarray[np.number], np.number]:
		yield self.node
		yield self.distance

class KDTree:
	def __init__(self, df: pd.DataFrame, k: int = 3) -> Self:
		nodes: np.ndarray[np.ndarray[np.number]] = df.to_numpy()
		self.dimension: int = df.shape[1] - 1
		assert self.dimension > 0
		def partition(arr, low: int, high: int, dim: int) -> int:
			pivot: np.number = arr[high, dim]
			i: int = low - 1

			for j in range(low, high):
				if arr[j, dim] <= pivot:
					i += 1
					arr[[i, j]] = arr[[j, i]]

			arr[[i + 1, high]] = arr[[high, i + 1]]
			return i + 1

		def quickselect(arr, low, high, k, dim) -> int:
			while low < high:
				pi: int = partition(arr, low, high, dim)
				if pi == k:
					return pi
				elif pi > k:
					high = pi - 1
				else:
					low = pi + 1
			return low
		
		row_len: int = nodes.shape[0]

		self.level: int = math.floor(math.log2(row_len)) + 1
		self.nodes: List[Node] = [None for i in range(2 ** self.level - 1)]
		res = self.nodes
		queue: deque[tuple[int, np.ndarray[np.ndarray[np.number]], int]] = deque([(0, nodes, 0)])
		while queue:
			index: int
			part: np.ndarray[np.ndarray[np.number]]
			dim: int
			index, part, dim = queue.popleft()
			if part.shape[0] > 0: 
				high: int = part.shape[0] - 1
				mid_index: int = quickselect(part, 0, high, high // 2, dim)
				res[index] = Node(part[mid_index][:-1].copy(), int(part[mid_index][-1]))
				if mid_index > 0:
					left: np.ndarray[np.ndarray[np.number]] = part[:mid_index]
					queue.append((2 * index + 1, left, (dim + 1) % self.dimension))
				if mid_index + 1 < part.shape[0]:
					right: np.ndarray[np.ndarray[np.number]] = part[mid_index + 1:]
					queue.append((2 * index + 2, right, (dim + 1) % self.dimension))
		del queue
		self.heap = []

	def print(self, level: int | None = None) -> None:
		node_count: int = 1
		if level is None:
			level = self.level
		for i in range(0, level):
			st: int = 1 << i
			print(f"{i + 1}: {[(self.nodes[st - 1 + j].array.tolist(), self.nodes[st - 1 + j].category) if self.nodes[st - 1 + j] is not None else [] for j in range(0, node_count)]}")
			node_count: int = node_count << 1

	def traverse(self, curr: NodePair, dim: int, point: np.ndarray[np.number], index: int):
		if index >= len(self.nodes):
			return
		current_node: Node = self.nodes[index]
		if current_node == None:
			return
		
		node, distance = curr

		# Exclude the category from the distance measurement
		current_distance: np.float64 = np.linalg.norm(current_node.array - point)

		# Push the element into the heap
		if len(self.heap) >= self.__k:
			heappushpop(self.heap, NodePair(current_node, current_distance))
		else:
			heappush(self.heap, NodePair(current_node, current_distance))

		if current_distance < distance:
			node = current_node
			distance = current_distance
			curr = NodePair(node=node, distance=distance)

		# Determine which branch to traverse first
		if point[dim] <= current_node.array[dim]:
			near_index = 2 * index + 1
			far_index = 2 * index + 2
		else:
			near_index = 2 * index + 2
			far_index = 2 * index + 1

		# Traverse the near branch
		self.traverse(curr, (dim + 1) % self.dimension, point, near_index)

		# Check if we need to traverse the far branch
		if abs(point[dim] - current_node.array[dim]) <= distance:
			self.traverse(curr, (dim + 1) % self.dimension, point, far_index)

	def get_k_nearest_neighbors(self, point: np.ndarray[np.number], k: int | None = None) -> List[np.ndarray[np.number]]:
		assert point.shape[0] == self.dimension, f"Incompatible data for tree with dimension {self.dimension} (got {point.shape})"
		if k is not None:
			self.__k = k
		self.traverse(NodePair(node=None, distance=np.inf), 0, point, 0)
		assert len(self.heap) > 0, "unreachable"
		ret, self.heap = np.array(self.heap), []
		return ret
		
		
if __name__ == "__main__":
	a = pd.DataFrame({ "temperature":np.array([1,2,3,4,8,7,6,5,9,10]), "wind_speed":np.array([-1,-2,-3,-4,-5,-6,-7,-8,-9,-10]), 
				  "humidity":np.array([11,12,13,14,15,16,17,18,19,20]), "precipitation":np.array([-11,-12,-13,-14,-15,-16,-17,-18,-19,-20]), 
				   "category":np.array([0,0,0,0,1,1,1,1,2,2])})
	t = KDTree(a)
	t.print()