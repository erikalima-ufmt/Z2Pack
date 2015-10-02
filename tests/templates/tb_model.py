#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>
# Date:    15.10.2014 10:22:43 CEST
# File:    tb_example.py

from common import *

import os
import types
import shutil

class TbModelTestCase(BuildDirTestCase):

    def create_model(self, t1, t2):
        on_site = [1, 1, -1, -1]
        hoppings = [[0, 2, [0, 0, 0], t1],
                    [0, 2, [-1, 0, 0], t1 * (-1j)],
                    [0, 2, [-1, -1, 0], t1 * (-1)],
                    [0, 2, [0, -1, 0], t1 * (1j)],
                    [1, 3, [0, 0, 0], t1],
                    [1, 3, [-1, 0, 0], t1 * (1j)],
                    [1, 3, [-1, -1, 0], t1 * (-1)],
                    [1, 3, [0, -1, 0], t1 * (-1j)],
                    [0, 0, [1, 0, 0], t2],
                    [0, 0, [0, 1, 0], t2],
                    [1, 1, [1, 0, 0], t2],
                    [1, 1, [0, 1, 0], t2],
                    [2, 2, [1, 0, 0], -t2],
                    [2, 2, [0, 1, 0], -t2],
                    [3, 3, [1, 0, 0], -t2],
                    [3, 3, [0, 1, 0], -t2]]
                    
        positions = [[0., 0., 0.], [0., 0., 0.], [0.5, 0.5, 0.], [0.5, 0.5, 0.]]
        self.model = z2pack.em.tb.Model(on_site, hoppings, positions, occ=2)

    # this test may produce false negatives due to small numerical differences
    def test_res1(self):
        self.create_model(0.2, 0.3)
        # call to Z2Pack
        tb_system = z2pack.em.tb.System(self.model)
        tb_surface = tb_system.surface(lambda kx, ky: [kx / 2, ky, 0], pickle_file=None)
        tb_surface.wcc_calc(verbose=False, num_strings=20)

        res = in_place_replace(tb_surface.get_res())

        self.assertFullAlmostEqual(tb_surface.get_res(), res)

    def test_res2(self):
        """ test pos_check=False """
        self.create_model(0, 0.3)
        # call to Z2Pack
        tb_system = z2pack.em.tb.System(self.model)
        tb_surface = tb_system.surface(lambda kx, ky: [kx / 2, ky, 0], pickle_file=None)
        tb_surface.wcc_calc(verbose=False,
                            num_strings=20,
                            pos_tol=None)

        res = in_place_replace(tb_surface.get_res())

        self.assertFullAlmostEqual(tb_surface.get_res(), res)

    def test_res3(self):
        """ test gap_tol=None """
        self.create_model(0.1, 0.3)
        # call to Z2Pack
        tb_system = z2pack.em.tb.System(self.model)
        tb_surface = tb_system.surface(lambda kx, ky: [kx / 2, ky, 0], pickle_file=None)
        tb_surface.wcc_calc(verbose=False,
                            num_strings=20,
                            gap_tol=None)

        res = in_place_replace(tb_surface.get_res())

        self.assertFullAlmostEqual(tb_surface.get_res(), res)

    def test_res4(self):
        """ test move_tol=None """
        self.create_model(0.1, 0.3)
        # call to Z2Pack
        tb_system = z2pack.em.tb.System(self.model)
        tb_surface = tb_system.surface(lambda kx, ky: [kx / 2, ky, 0], pickle_file=None)
        tb_surface.wcc_calc(verbose=False,
                          num_strings=20,
                          move_tol=None)

        res = in_place_replace(tb_surface.get_res())

        self.assertFullAlmostEqual(tb_surface.get_res(), res)

    def test_res5(self):
        """ test gap_tol=None and move_tol=None"""
        self.create_model(0.1, 0.3)
        # call to Z2Pack
        tb_system = z2pack.em.tb.System(self.model)
        tb_surface = tb_system.surface(lambda kx, ky: [kx / 2, ky, 0], pickle_file=None)
        tb_surface.wcc_calc(verbose=False,
                            num_strings=20,
                            gap_tol=None,
                            move_tol=None)

        
        res = in_place_replace(tb_surface.get_res())

        self.assertFullAlmostEqual(tb_surface.get_res(), res)

    def test_change_uc(self):
        """test chaning the unit cell"""
        self.create_model(0.1, 0.3)
        model = self.model.change_uc([[1, 1, 1], [0, 1, 1], [0, 0, 1]])
        # call to Z2Pack
        tb_system = z2pack.em.tb.System(model)
        tb_surface = tb_system.surface(lambda kx, ky: [kx / 2, ky, 0], pickle_file=None)
        tb_surface.wcc_calc(verbose=False,
                            num_strings=20,
                            gap_tol=None,
                            move_tol=None)

        
        res = in_place_replace(tb_surface.get_res())

        self.assertFullAlmostEqual(tb_surface.get_res(), res)
        
    def test_change_uc_inplace(self):
        """test chaning the unit cell"""
        self.create_model(0.1, 0.3)
        model = self.model.change_uc([[1, 1, 1], [0, 1, 1], [0, 0, 1]])
        self.model.change_uc([[1, 1, 1], [0, 1, 1], [0, 0, 1]], in_place=True)
        system0 = z2pack.em.tb.System(model)
        surface0 = system0.surface(lambda kx, ky: [kx / 2, ky, 0], pickle_file=None)
        surface0.wcc_calc(verbose=False,
                          num_strings=20,
                          gap_tol=None,
                          move_tol=None)
        system1 = z2pack.em.tb.System(self.model)
        surface1 = system1.surface(lambda kx, ky: [kx / 2, ky, 0], pickle_file=None)
        surface1.wcc_calc(verbose=False,
                          num_strings=20,
                          gap_tol=None,
                          move_tol=None)
        
        self.assertFullAlmostEqual(surface0.get_res(), surface1.get_res())
        
    def test_change_uc_error(self):
        """test chaning the unit cell"""
        self.create_model(0.1, 0.3)
        self.assertRaises(ValueError, self.model.change_uc, [[2, 1, 1], [0, 1, 1], [0, 0, 1]])

    def test_supercell1(self):
        """test creating a supercell"""
        self.create_model(0.2, 0.3)
        model = self.model.supercell([1, 1, 2], [True, True, False])
        # call to Z2Pack
        tb_system = z2pack.em.tb.System(model)
        tb_surface = tb_system.surface(lambda kx, ky: [kx / 2, ky, 0], pickle_file=None)
        tb_surface.wcc_calc(verbose=False,
                            num_strings=5,
                            gap_tol=None,
                            move_tol=None)

        
        res = in_place_replace(tb_surface.get_res())

        self.assertFullAlmostEqual(tb_surface.get_res(), res)

    def test_supercell_inplace(self):
        """test creating a supercell"""
        self.create_model(0.2, 0.3)
        model = self.model.supercell([1, 1, 2], [True, True, False])
        self.model.supercell([1, 1, 2], [True, True, False], in_place=True)
        system0 = z2pack.em.tb.System(model)
        surface0 = system0.surface(lambda kx, ky: [kx / 2, ky, 0], pickle_file=None)
        surface0.wcc_calc(verbose=False,
                          num_strings=20,
                          gap_tol=None,
                          move_tol=None)
        system1 = z2pack.em.tb.System(self.model)
        surface1 = system1.surface(lambda kx, ky: [kx / 2, ky, 0], pickle_file=None)
        surface1.wcc_calc(verbose=False,
                          num_strings=20,
                          gap_tol=None,
                          move_tol=None)
        
        self.assertFullAlmostEqual(surface0.get_res(), surface1.get_res())
        
    def test_supercell2(self):
        """test creating a supercell"""
        self.create_model(0.2, 0.3)
        model = self.model.supercell([1, 1, 3], [True, True, True])
        # call to Z2Pack
        tb_system = z2pack.em.tb.System(model)
        tb_surface = tb_system.surface(lambda kx, ky: [kx / 2, ky, 0], pickle_file=None)
        tb_surface.wcc_calc(verbose=False,
                            num_strings=5,
                            gap_tol=None,
                            move_tol=None,
                            pos_tol=None)

        
        res = in_place_replace(tb_surface.get_res())

        self.assertFullAlmostEqual(tb_surface.get_res(), res)
        
    def test_supercell3(self):
        """test creating a supercell"""
        self.create_model(0.2, 0.3)
        model = self.model.supercell([2, 1, 3], [True, True, True])
        # call to Z2Pack
        tb_system = z2pack.em.tb.System(model)
        tb_surface = tb_system.surface(lambda kx, ky: [kx / 2, ky, 0], pickle_file=None)
        tb_surface.wcc_calc(verbose=False,
                            num_strings=5,
                            gap_tol=None,
                            move_tol=None)

        
        res = in_place_replace(tb_surface.get_res())

        self.assertFullAlmostEqual(tb_surface.get_res(), res)
        
    def test_supercell4(self):
        """test creating a supercell"""
        self.create_model(0.2, 0.3)
        model = self.model.supercell([1, 1, 1], [False, False, False])
        # call to Z2Pack
        tb_system = z2pack.em.tb.System(model)
        tb_surface = tb_system.surface(lambda kx, ky: [kx / 2, ky, 0], pickle_file=None)
        tb_surface.wcc_calc(verbose=False,
                            num_strings=5,
                            gap_tol=None,
                            move_tol=None)

        
        res = in_place_replace(tb_surface.get_res())

        self.assertFullAlmostEqual(tb_surface.get_res(), res)

    def test_precompute(self):
        """ test that precomputation will be redone after changing _hop or _on_site"""
        self.create_model(0.1, 0.3)
        self.model._unchanged = True
        self.model._hop = []
        self.assertEqual(self.model._unchanged, False)
        self.model._unchanged = True
        self.model._on_site = []
        self.assertEqual(self.model._unchanged, False)

    def test_warning(self):
        """ test the warning that is given when string_vec != None"""
        self.create_model(0.1, 0.3)
        # call to Z2Pack
        tb_system = z2pack.em.tb.System(self.model)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            tb_surface = tb_system.surface(lambda kx: [kx / 2, 0, 0], [0, 1, 0], pickle_file=None)
            assert len(w) == 1
            assert w[-1].category == DeprecationWarning
            assert "string_vec" in str(w[-1].message)

    def test_saveload(self):
        self.create_model(0.1, 0.3)
        tb_system = z2pack.em.tb.System(self.model)
        surface1 = tb_system.surface(lambda kx, ky: [kx / 2, ky, 0], pickle_file=self._build_folder + '/tb_pickle.txt')
        surface2 = tb_system.surface(lambda kx, ky: [kx / 2, ky, 0], pickle_file=self._build_folder + '/tb_pickle.txt')
        surface1.wcc_calc(verbose=False)
        surface2.load()
        self.assertFullAlmostEqual(surface1.get_res(), surface2.get_res())

    def testkwargcheck1(self):
        """ test kwarg check on wcc_calc """
        self.create_model(0.1, 0.3)
        # call to Z2Pack
        tb_system = z2pack.em.tb.System(self.model)
        tb_surface = tb_system.surface(lambda kx, ky: [kx / 2, ky, 0], pickle_file=None)
        self.assertRaises(
            TypeError,
            tb_surface.wcc_calc,
            invalid_kwarg = 3)

    def testkwargcheck2(self):
        """ test kwarg check on __init__ """
        self.create_model(0, 0.3)
        # call to Z2Pack
        tb_system = z2pack.em.tb.System(self.model)
        self.assertRaises(
            TypeError,
            tb_system.surface,
            1, 2, 0, invalid_kwarg = 3)

if __name__ == "__main__":
    unittest.main()
