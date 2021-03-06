#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>
# Date:    15.10.2014 14:49:25 CEST
# File:    fp_phony.py

from common import *

import os
import re
import types
import shutil
import unittest
import platform


class BiAbinitTestCase(AbinitTestCase):
    def __init__(self, *args, **kwargs):
        self._input_folder = 'samples/abinit'

        self._input_files = [self._input_folder + '/' + name for name in
                             ['Bi_nscf.files', 'Bi_nscf.in', 'wannier90.win', '83bi.5.hgh', 'Bi_scf_o_DEN']]

        super(BiAbinitTestCase, self).__init__(*args, **kwargs)

        self.system = z2pack.fp.System(
            self._input_files,
            z2pack.fp.kpts.abinit,
            "Bi_nscf.in",
            "mpirun -np 4 abinit < Bi_nscf.files >& log",
            executable='/bin/bash',
            build_folder=self._build_folder)

    def test_bismuth_0(self):
        surface = self.system.surface(lambda kx: [0, kx / 2., 0], [0, 0, 1], pickle_file=None)
        surface.wcc_calc(pos_tol=None, gap_tol=None, verbose=False, num_strings=4)

        wcc = in_place_replace(surface.get_res()['wcc'])
        t_par = in_place_replace(surface.get_res()['t_par'])
        self.assertWccConv(wcc, surface.get_res()['wcc'])
        self.assertFullAlmostEqual(t_par, surface.get_res()['t_par'])

    def test_bismuth_1(self):
        surface = self.system.surface(lambda kx: [0, kx, kx], [1, 0, 0], pickle_file=None)

        surface.wcc_calc(pos_tol=None, gap_tol=None, verbose=False, num_strings=4)

        wcc = in_place_replace(surface.get_res()['wcc'])
        t_par = in_place_replace(surface.get_res()['t_par'])
        self.assertWccConv(wcc, surface.get_res()['wcc'])
        self.assertFullAlmostEqual(t_par, surface.get_res()['t_par'])

if __name__ == "__main__":
    unittest.main()
