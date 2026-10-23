import unittest
import numpy as np
from engine.tensor import Tensor

class TestAutodiffEngine(unittest.TestCase):
    def test_addition_backward(self):
        a = Tensor([2.0], requires_grad=True)
        b = Tensor([3.0], requires_grad=True)
        c = a + b
        c.backward()
        
        np.testing.assert_array_equal(a.grad, np.array([1.0]))
        np.testing.assert_array_equal(b.grad, np.array([1.0]))

    def test_multiplication_backward(self):
        a = Tensor([2.0], requires_grad=True)
        b = Tensor([3.0], requires_grad=True)
        c = a * b
        c.backward()
        
        np.testing.assert_array_equal(a.grad, np.array([3.0]))
        np.testing.assert_array_equal(b.grad, np.array([2.0]))

    def test_relu_backward(self):
        a = Tensor([-2.0, 3.0], requires_grad=True)
        b = a.relu()
        b.backward()
        
        np.testing.assert_array_equal(a.grad, np.array([0.0, 1.0]))

if __name__ == '__main__':
    unittest.main()