import unittest
from relativisticpy.core.indices import Idx, Indices

class TestIdx(unittest.TestCase):

    def setUp(self):
        self.idx1 = Idx('a', order=1, values=5, covariant=True)
        self.idx2 = Idx('a', order=1, values=5, covariant=False)
        self.idx3 = Idx('b', order=2, values=4, covariant=True)

    def test_negation(self):
        self.assertEqual(-Idx('a', order=1, values=5, covariant=True), Idx('a', order=1, values=5, covariant=False))

    def test_equality(self):
        self.assertNotEqual(Idx('a'), Idx('b')); self.assertEqual(Idx('a'), Idx('a')); self.assertNotEqual(Idx('a'), -Idx('a')) # Obvious checks
        self.assertEqual(Idx('a', order=1, values=5, covariant=True), Idx('a', order=1, values=5, covariant=True)) # All the same
        self.assertEqual(Idx('a', order=1, values=5, covariant=True), Idx('a', order=2, values=5, covariant=True)) # Order should not matter
        self.assertEqual(Idx('a', order=1, values=5, covariant=True), Idx('a', order=2, values=6, covariant=True)) # Values should not matter
        
    def test_is_contracted_with(self):
        self.assertTrue(Idx('a', covariant=True).is_contracted_with(Idx('a', covariant=False))) # Symbol == Symbol and cov != cov => summed
        self.assertFalse(Idx('a', covariant=True).is_contracted_with(Idx('a', covariant=True))) # Equal covariance => not summed
        self.assertFalse(Idx('b', covariant=True).is_contracted_with(Idx('a', covariant=False))) # Different symbol => not summed

    def test_iteration(self): self.fail('Test not implemented.')
    def test_len(self): self.fail('Test not implemented.')
    def test_repr(self): self.fail('Test not implemented.')
    def test_str(self): self.fail('Test not implemented.')
    def test_is_identical_to(self): self.fail('Test not implemented.')
    def test_is_summed_wrt_indices(self): self.fail('Test not implemented.')
    def test_get_summed_location(self): self.fail('Test not implemented.')
    def test_get_repeated_location(self): self.fail('Test not implemented.')
    def test_get_summed_locations(self): self.fail('Test not implemented.') # Repeated method <- remove
    def test_get_repeated_locations(self): self.fail('Test not implemented.') # Repeated method <- remove

