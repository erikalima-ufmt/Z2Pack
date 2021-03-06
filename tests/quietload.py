#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>
# Date:    10.03.2015 10:53:12 CET
# File:    quietload.py

from common import *

import os
import types
import shutil

class QuietLoadTestCase(CommonTestCase):
    def __init__(self, *args, **kwargs):
        try:
            shutil.rmtree('build/quietload')
        except OSError:
            pass
        os.mkdir('build/quietload')
        super(QuietLoadTestCase, self).__init__(*args, **kwargs)
        
    def test_q(self):
        system = z2pack.System(lambda kx, N: [])
        surface = system.surface(lambda kx: [0, 0, kx], [1, 0, 0], pickle_file='build/quietload/dummy.txt')
        surface.load(quiet=True)
        
    def test_nq(self):
        system = z2pack.System(lambda kx, N: [])
        surface = system.surface(lambda kx: [0, 0, kx], [1, 0, 0], pickle_file='build/quietload/dummy.txt')
        self.assertRaises(IOError, surface.load)
        
if __name__ == "__main__":
    unittest.main()
    
