import unittest
from relativisticpy.workbook.workbook import Workbook

class TestIndices(unittest.TestCase):

    def setUp(self):
        self.wb = Workbook()

        self.file_one_path = './test_files/basic_calculus_example.txt'
        self.file_two_path = './test_files/black_hole.txt'
        self.file_three_path = './test_files/Schild_solution.txt'

        self.one_result = 'res a'
        self.two_result = 'res b'
        self.three_result = 'res c'

    # Dunder Tests
    def test_one(self): self.assertEqual(self.wb.exe(self.file_one_path), self.one_result)
    def test_two(self): self.assertEqual(self.wb.exe(self.file_two_path), self.one_result)
    def test_three(self): self.assertEqual(self.wb.exe(self.file_three_path), self.one_result)
