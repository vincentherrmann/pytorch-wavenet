from unittest import TestCase
from scipy.io import wavfile
import torch

from model import Dilated_queue


class Test_dilated_queue(TestCase):
	def test_enqueue(self):
		queue = Dilated_queue(max_length=8, num_channels=3)
		e = torch.zeros((3))
		for i in range(11):
			e = e + 1
			queue.enqueue(e)

		data = queue.data[0, :]
		print('data: ', data)
		assert data[0] == 9
		assert data[2] == 11
		assert data[7] == 8


	def test_dequeue(self):
		queue = Dilated_queue(max_length=8, num_channels=1)
		e = torch.zeros((1))
		for i in range(11):
			e = e + 1
			queue.enqueue(e)

		print('data: ', queue.data)

		for i in range(9):
			d = queue.dequeue(num_deq=3, dilation=2)
			print(d)

		assert d[0][0] == 5
		assert d[0][1] == 7
		assert d[0][2] == 9

class Test_wav_files(TestCase):
	def test_wav_read(self):
		data = wavfile.read('trained_generated.wav')[1]
		print(data)
		#[0.1, -0.53125...
		assert False



