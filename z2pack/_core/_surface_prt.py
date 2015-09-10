#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@gmx.ch>
# Date:    16.02.2015 09:27:19 CET
# File:    output.py

r"""
Handles verbose printing for the Surface class.
"""

from __future__ import print_function

from ..ptools import string_tools

import sys
import time
import numpy as np

class PrintFunctions(object):
    r"""
    Contains 'decorator' functions that carry the same name as the Surface method they decorate. This class acts as a namespace.
    """
    def _call_line(func):
        def inner(self, i, t):
            # initial output
            _print(self, "Calculating string at t = {0:.4f}, k = {1}:\n".
                   format(t, string_tools.fl_to_s(self._param_fct(t, 0.))))
            #-----------------------------------------------------------#
            conv, changed = func(self, i, t)
            #-----------------------------------------------------------#
            # check convergence flag
            if not conv:
                self._log.log('pos check', t, string_tools.fl_to_s(self._param_fct(t, 0.)))

            return changed # cut out convergence flag
        return inner

    def _check_neighbours(func):
        def inner(self):
            res = func(self)
            if res is None:
                _print(self, 'Skipping neighbour checks (gap check & move check).\n\n')
            return res
        return inner

    def _check_single_neighbour(func):
        def inner(self, i):
            _print(self, ('Checking neighbouring t-points t = {0:.4f} and ' +
                          't = {1:.4f}\n').format(self._t_points[i], self._t_points[i + 1]))
            #-----------------------------------------------------------#
            res = func(self, i)
            #-----------------------------------------------------------#
            if all(res):
                _print(self, "Condition fulfilled!\n\n")
            else:
                if not res[0]:
                    _print(self, 'Gap check not fulfilled yet.\n')
                if not res[1]:
                    _print(self, 'Move check not fulfilled yet.\n')
            return res
        return inner

    def _add_string(func):
        def inner(self, i):
            res = func(self, i)
            if res:
                _print(self, 'Added string at t = {0}\n\n'.format(self._t_points[i + 1]))
            else:
                _print(self, 'Reached minimum distance between neighbours\n\n')
            return res
        return inner
        
    def _add_string_at(func):
        def inner(self, t):
            res = func(self, t)
            string = '{0:<5}'.format('{0:.5f}'.format(t).rstrip('0'))
            if res:
                _print(self, 't = {0} added\n'.format(string))
            else:
                _print(self, 't = {0} exists\n'.format(string))
            return res
        return inner
        
    def _var_refresh(func):
        def inner(self):
            _print(self, 'Setting up strings:\n-------------------\n')
            func(self)
            _print(self, '\nMain calculation\n----------------\n')
        return inner

    def _wcc_calc_main(func):
        def inner(self):
            #----------------initial output-----------------------------#
            start_time = time.time()
            string = "starting wcc calculation\n\n"
            length = max(len(key) for key in self._current.keys()) + 2
            for key in sorted(self._current.keys()):
                value = str(self._current[key])
                if len(value) > 48:
                    value = value[:45] + '...'
                string += key.ljust(length) + value + '\n'
            string = string[:-1]
            _print(self, string_tools.cbox(string) + '\n\n')
            
            #----------------computation--------------------------------#
            res = func(self)
            #----------------final output-------------------------------#
            end_time = time.time()
            duration = end_time - start_time
            duration_string = str(int(np.floor(duration / 3600))) + \
                " h " + str(int(np.floor(duration / 60)) % 60) + \
                " min " + str(int(np.floor(duration)) % 60) + " sec"
            _print(self,
                   string_tools.cbox(
                       ["finished wcc calculation" + "\ntime: " + duration_string,
                        'CONVERGENCE REPORT\n------------------\n\n' +
                        str(self._log)]) +
                   '\n')
            return res
        return inner

def _print(self, string):
    if self._current['verbose']:
        print(string, end='')
        sys.stdout.flush()

def prt_dispatcher(func):
    return PrintFunctions.__dict__[func.__name__](func)
