#!/usr/bin/env python3

import numpy as np
import random
import math
import time
import _matrix


class TestMatrix:
    def is_close_to_equal(self, m1, m2, m, n):
        for i in range(m):
            for j in range(n):
                assert math.fabs(m1[i, j] - m2[i, j]) < 0.00000001
        return True

    def time_report(self, t):
        if t == -1:
            return time.process_time()
        else:
            return time.process_time() - t

    def calculation_helper(self, p, q, r, tsize):
        m1 = np.random.rand(p, q)
        m2 = np.random.rand(q, r)
        m3 = np.matmul(m1, m2)

        _m1 = _matrix.Matrix(m1)
        _m2 = _matrix.Matrix(m2)

        start = self.time_report(-1)
        naive = _matrix.multiply_naive(_m1, _m2)
        tnaive = self.time_report(start)

        start = self.time_report(-1)
        tile = _matrix.multiply_tile(_m1, _m2, tsize)
        ttile = self.time_report(start)

        start = self.time_report(-1)
        mkl = _matrix.multiply_mkl(_m1, _m2)
        tmkl = self.time_report(start)

        assert m3.shape[0] == naive.nrow and m3.shape[1] == naive.ncol
        assert m3.shape[0] == tile.nrow and m3.shape[1] == tile.ncol
        assert m3.shape[0] == mkl.nrow and m3.shape[1] == mkl.ncol

        assert (self.is_close_to_equal(m3, naive, p, r))
        assert (self.is_close_to_equal(m3, tile, p, r))
        assert (self.is_close_to_equal(m3, mkl, p, r))

        f = open("performance.txt", "w")
        f.write('multiply_naive costs: {} seconds\n'.format(tnaive))
        f.write('multiply_tile costs: {} seconds\n'.format(ttile))
        f.write('multiply_mkl costs: {} seconds\n'.format(tmkl))
        f.write('tile speed-up over naive: {}\n'.format(tnaive / ttile))
        f.write('MKL speed-up over naive: {}\n'.format(tnaive / tmkl))
        f.close()

    def test_large(self):
        self.calculation_helper(1000, 1000, 1000, 24)

    def test_random(self):
        last = 1
        while last < 10:
            p = random.randint(1, 1000)
            q = random.randint(1, 1000)
            r = random.randint(1, 1000)
            tsize = random.randint(1, 1000)
            self.calculation_helper(p, q, r, tsize)
            last = last + 1