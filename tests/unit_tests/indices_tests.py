import unittest
from relativisticpy.core.indices import Idx, Indices

class TestIndices(unittest.TestCase):

    def setUp(self):
        self.idx1 = Idx('a', order=1, values=5, covariant=True)
        self.idx2 = Idx('b', order=2, values=4, covariant=True)
        self.indices1 = Indices(self.idx1, self.idx2)
        self.indices2 = Indices(self.idx1, -self.idx2)

    # Dunder Tests
    def test_len(self): self.assertEqual(len(self.indices1), 2)
    def test_list(self): self.assertEqual(list(self.indices1), [self.idx1, self.idx2])
    def test_equality(self): self.assertEqual(self.indices1, self.indices1); self.assertNotEqual(self.indices1, self.indices2)
    def test_getitem(self): self.assertEqual(self.indices1[0], self.idx1); self.assertEqual(self.indices1[1], self.idx2)
    def test_getitem(self): self.assertEqual(self.indices1[0], self.idx1); self.assertEqual(self.indices1[1], self.idx2)
    def test_str(self): self.assertEqual(str(self.indices1), "a^1 b^2")
    def test_repr(self): self.assertEqual(repr(self.indices1), "Indices(Idx('a', order=1, values=5, covariant=True), Idx('b', order=2, values=4, covariant=True))")
    def test_add(self): indices3 = self.indices1 + self.indices2; self.assertIsInstance(indices3, Indices)
    def test_mul(self): indices3 = self.indices1 * self.indices2; self.assertIsInstance(indices3, Indices)
    def test_sub(self): indices3 = self.indices1 - self.indices2; self.assertIsInstance(indices3, Indices)
    def test_index(self): self.fail('Test not implemented.')

    def test_self_product(self):
        self.assertFalse(self.indices1.self_product().self_summed)
        self.assertTrue(self.indices2.self_product().self_summed)

    def test_contractions(self): self.assertEqual(self.indices1.contractions(self.indices2), 1)

    # # Dunders
    # def __getitem__(self, index: Idx) -> List[Idx]: return [idx for idx in self.indices if idx.symbol == index.symbol and idx.covariant == index.covariant]
    # def __setitem__(self, key: Idx, new: Idx) -> 'Indices': return Indices(*[new if idx.symbol == key.symbol and idx.covariant == key.covariant else idx for idx in self.indices])
