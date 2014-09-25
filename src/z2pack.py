#!/usr/bin/python3.2
# -*- coding: utf-8 -*-
#
# Author:  Dominik Gresch <greschd@ethz.ch>
# Date:    21.03.2014 11:46:38 CET
# File:    z2pack.py

from __future__ import print_function

import python_tools.string_tools as string_tools

# for the ABINIT specialisation
import abinit.abinit_run as ar
import abinit.abinit_input_io as io

# for the tight-binding specialisation
import tight_binding.tb_vectors as TbVectors 
from tight_binding.tight_binding import TbSystem

import sys
import time
import copy
import pickle
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
#                           LIBRARY CORE                                #
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
class Z2PackSystem:
    """
    Z2PackSystem Class
    ~~~~~~~~~~~~~~~~~~
    abstract Base Class for Z2Pack systems (Interface definition)
    method: plane
    """
    
    def __init__(self, M_handle_creator, **kwargs):
        """
        M_handle_creator: takes (string_dir, plane_pos_dir, plane_pos)
            and creates an M_handle
            --> M_handle(kx, N): creates M-matrices
        
        kwargs: passed to Z2PackPlane constructor unless overwritten
            by plane() kwargs
        """
        self._defaults = kwargs
        self._M_handle_creator = M_handle_creator
    

    def plane(self, string_dir, plane_pos_dir, plane_pos, **kwargs):
        """
        plane position: orthogonal to plane_pos_dir, at plane_pos
        strings: along string_dir
        
        kwargs: passed to Z2PackPlane constructor. Take precedence
            over kwargs from Z2PackSystem constructor.
        """
        # updating keyword arguments
        kw_arguments = copy.copy(self._defaults)
        kw_arguments.update(kwargs)
        
        # creating M_handle
        if(string_dir == plane_pos_dir):
            raise ValueError('strings cannot be perpendicular to the plane')
        
        return Z2PackPlane(   
                                    self._M_handle_creator(string_dir, plane_pos_dir, plane_pos),
                                    **kw_arguments
                                )

class Z2PackPlane:
    """
    Z2PackPlane Class:
    ~~~~~~~~~~~~~~~~~
    specifies a plane in the 3D system, on which to calculate the 
    topological invariant. This is achieved via the M_handle input
    variable. 
    The M_handle is created by Z2PackSystem.plane(), and as such is 
    specific to the type of Z2Pack calculation (ABINIT, Tb, ...)
    
    methods: wcc_calc, load, plot, invariant
    """
    
    def __init__(   self, 
                    M_handle, 
                    pickle_file = "res_pickle.txt", 
                    **kwargs):
        """
        constructor
        parses the input variables
        
        args:
        ~~~~
        M_handle:           should create a list of MMN given (k, N)
        pickle_file:        path to file for saving using pickle
        
        kwargs:
        ~~~~~~
        same as for wcc_calc. kwargs for wcc_calc take precedence
        """
        self._M_handle = M_handle
        self._pickle_file = pickle_file
        self._defaults = {  'wcc_tol': 1e-2,
                            'gap_tol': 2e-2,
                            'max_iter': 10,
                            'use_pickle': True,
                            'Nstrings': 11,
                            'verbose': True
                            }
        self._defaults.update(kwargs)
        
    def wcc_calc(self, **kwargs):
        """
        calculating the wcc of the system
        - automated convergence in string direction
        - automated check for distance between gap and wcc -> add string
        
        kwargs:
        ~~~~~~
        wcc_tol:            maximum movement of wcc between two steps 
                            for convergence
        gap_tol:            minimum size of gap between the largest gap
                            and the closest wcc at neighbouring strings
        max_iter:           max. number of iterations for 1 string
        use_pickle:         toggles use of pickle for saving
        Nstrings:           number of strings at the beginning (should be 
                            >= 8 for good results)
        verbose:            toggles output printed
        """
        kwarguments = copy.copy(self._defaults)
        kwarguments.update(kwargs)
        # parse input variables
        self._wcc_tol = kwarguments['wcc_tol']
        self._gap_tol = kwarguments['gap_tol']
        self._max_iter = kwarguments['max_iter']
        self._use_pickle = kwarguments['use_pickle']
        self._Nstrings = kwarguments['Nstrings']
        self._verbose = kwarguments['verbose']
            
        # checking Nstrings
        if(self._Nstrings < 2):
            raise ValueError("Nstrings must be at least 2")
        elif(self._Nstrings < 8):
            warnings.warn("Nstrings should usually be >= 8 for good results", UserWarning)

        # initial output 
        if(self._verbose):
            print(string_tools.cbox( "starting wcc calculation\n\n" +\
                                "kwargs:\n" +\
                                "wcc_tol" + "     " + str(self._wcc_tol) + "\n" +\
                                "gap_tol" + "     " + str(self._gap_tol) + "\n" +\
                                "max_iter" + "    " + str(self._max_iter) + "\n" +\
                                "use_pickle" + "  " + str(self._use_pickle) + "\n" +\
                                "Nstrings" + "    " + str(self._Nstrings) 
                                ) + "\n")
            
        start_time = time.time()

        # initialising
        self._k_points = list(np.linspace(0, 0.5, self._Nstrings, endpoint = True))
        self._gaps = [None for i in range(self._Nstrings)]
        self._wcc_list = [[] for i in range(self._Nstrings)] 
        self._neighbour_check = [False for i in range(self._Nstrings - 1)]
        self._string_status = [False for i in range(self._Nstrings)]
        
                
        # main calculation part
        while not (all(self._neighbour_check)):
            for i, kx in enumerate(self._k_points):
                if not(self._string_status[i]):
                    self._wcc_list[i] = self._getwcc(kx)
                    self._gaps[i] = self._gapfind(self._wcc_list[i])
                    self._string_status[i] = True
                    self._save()
                    
            self._check_neighbours()

        # dump results into pickle file
        self._save()
            
            
        # output to signal end of wcc calculation
        end_time = time.time()
        duration = end_time - start_time
        duration_string =   str(int(np.floor(duration / 3600))) + \
                            " h " + \
                            str(int(np.floor(duration / 60)) % 60) + \
                             " min " + \
                            str(int(np.floor(duration)) % 60) + \
                            " sec"
        if(self._verbose):
            print(string_tools.cbox( "finished wcc calculation" + "\ntime: " 
                            + duration_string))
        
        # return value
        return [self._k_points, self._wcc_list]
            
    #-------------------------------------------------------------------#
    #                support functions for wcc                          #
    #-------------------------------------------------------------------#

    # checking distance gap-wcc
    def _check_neighbours(self):
        """
        checks the neighbour conditions, adds a value in k_points when
        they are not fulfilled
        - adds at most one k_point per run
        - returns Boolean: all neighbour conditions fulfilled <=> True
        """
        for i, status in enumerate(self._neighbour_check):
            if not(status):
                if(self._string_status[i] and self._string_status[i + 1]):
                    if(self._verbose):
                        print("Checking neighbouring k-points k = " + "%.4f" % self._k_points[i] + " and k = " + "%.4f" % self._k_points[i + 1] + "\n", end = "")
                        sys.stdout.flush()
                    if(self._check_single_neighbour(i, i + 1)):
                        if(self._verbose):
                            print("Condition fulfilled\n\n", end = "")
                            sys.stdout.flush()
                        self._neighbour_check[i] = True
                    else:
                        if(self._verbose):
                            print("Condition not fulfilled\n\n", end = "")
                            sys.stdout.flush()
                        # add entries to neighbour_check, k_point and string_status
                        self._neighbour_check.insert(i + 1, False)
                        self._string_status.insert(i + 1, False)
                        self._k_points.insert(i + 1, (self._k_points[i] + self._k_points[i+1]) / 2)
                        self._wcc_list.insert(i + 1, [])
                        self._gaps.insert(i + 1, None)
                        # check length of the variables
                        assert len(self._k_points) == len(self._wcc_list)
                        assert len(self._k_points) - 1 == len(self._neighbour_check)
                        assert len(self._k_points) == len(self._string_status)
                        assert len(self._k_points) == len(self._gaps)
                        return False
                else:
                    return False
        return True
        
    
    def _check_single_neighbour(self, i, j):
        """
        checks if the gap[i] is too close to any of the WCC in 
        wcc_list[j] and vice versa
        should be used with j = i + 1
        """
        return self._check_single_direction(self._wcc_list[j], self._gaps[i])
        
    def _check_single_direction(self, wcc, gap):
        """
        checks if gap is too close to any of the elements in wcc
        """
        for wcc_val in wcc:
            if(min(abs(1 + wcc_val - gap) % 1, abs(1 - gap + wcc_val) % 1) < self._gap_tol):
                return False
        return True
        
    # pickle: save and load
    def _save(self):
        """
        save k_points, wcc and gaps to pickle file
        only works if use_pickle = True & path to pickle_file exists
        """
        if(self._use_pickle):
            f = open(self._pickle_file, "wb")
            pickle.dump([self._k_points, self._wcc_list, self._gaps], f)
            f.close()
            
    def load(self):
        """
        load k_points, wcc and gaps from pickle file
        only works if pickle_file exists
        """
        f = open(self._pickle_file, "rb")
        [self._k_points, self._wcc_list, self._gaps] = pickle.load(f)
        f.close()
    
    # calculating one string
    def _getwcc(self, kx):
        """
        calculates WCC along a string by increasing the number of steps 
        (k-points) along the string until the WCC converge
        """
        # initial output
        if(self._verbose):
            print("calculating string at kx = " + "%.4f" % kx + "; N = ", end = "")
            sys.stdout.flush()

        # first two steps
        N = 8
        niter = 0
        if(self._verbose):
            print(str(N), end = "")
            sys.stdout.flush() # Output
        x, min_sv = self._trywcc(self._M_handle(kx, N))
        # iteration
        while(True):
            # larger steps for small min_sv (every second step)
            if(niter % 2 == 1 and min_sv < 0.5): 
                N += 4
            else:
                N += 2
            xold = copy.copy(x)
            if(self._verbose):
                # Output
                print(", " + str(N), end = "")
                sys.stdout.flush()
            x, min_sv = self._trywcc(self._M_handle(kx, N))
            niter += 1

            # break conditions
            if(self._convcheck(x, xold)): # success
                if(self._verbose):
                    print(" finished!\n\n", end = "")
                    sys.stdout.flush()
                break
            if(niter > self._max_iter): # failure
                if(self._verbose):
                    print("failed to converge!\n\n", end = "")
                    sys.stdout.flush()
                break

        return sorted(x)
    
    def _trywcc(self, allM):
        """
        Calculates the WCC from the MMN matrices
        """
        Gamma = np.eye(len(allM[0]))
        min_sv = 1
        for M in allM:
            [V,E,W] = la.svd(M)
            Gamma = np.dot(np.dot(V,W).conjugate().transpose(), Gamma)
            min_sv = min(min(E), min_sv)
        # getting the wcc from the eigenvalues of gamma
        [eigs, _] = la.eig(Gamma)
        if(self._verbose):
            if(self._verbose):
                print(" (" + "%.3f" % min_sv + ")", end= "")
                sys.stdout.flush()
        return [(1j * np.log(z) / (2 * np.pi)).real % 1 for z in eigs], min_sv
    

        
    # wcc convergence functions
    def _convcheck(self, x, y):
        """
        check convergence of wcc from x to y
        
        depends on: self._wcc_tol
                    roughly corresponds to the total 'movement' in WCC that
                    is tolerated between x and y
        """
        if(len(x) != len(y)):
            if(self._verbose):
                print("Warning: consecutive strings don't have the same amount of WCC")
            return False
        else:
            return self._convsum(x, y, self._wcc_tol) < 1 
    

    def _convsum(self, listA, listB, epsilon = 1e-2, N0 = 7):
        """
        helper function for _convcheck
        
        calculates the absolute value of the change in density from listA
        to listB, when each WCC corresponds to a triangle of width epsilon
        (and total density = 1)
        """
        N = N0 * int(1/(2 * epsilon))
        val = np.zeros(N)
        for x in listA:
            index = int(N*x)
            for i in range(0, N0):
                val[(index - i) % N] += 1 - (i/N0)
            for i in range(1, N0):
                val[(index + i) % N] += 1 - (i/N0)
        for x in listB:
            index = int(N*x)
            for i in range(0, N0):
                val[(index - i) % N] -= 1 - (i/N0)
            for i in range(1, N0):
                val[(index + i) % N] -= 1 - (i/N0)
        return sum(abs(val)) / N0
        
    def _gapfind(self, x):
        """
        finds the largest gap in vector x, modulo 1
        """
        x = sorted(x)
        gapsize = 0
        gappos = 0
        N = len(x)
        for i in range(0, N - 1):
            temp = x[i + 1] - x[i]
            if(temp > gapsize):
                gapsize = temp
                gappos = i
        temp = x[0] - x[-1] + 1
        if(temp > gapsize):
            gapsize = temp
            gappos = N - 1
        return (x[gappos] + gapsize / 2) % 1
    #----------------END OF SUPPORT FUNCTIONS---------------------------#        
    
    def plot(self, shift = 0):
        """
        plot WCC and largest gaps (with a shift modulo 1)
        """
        shift = shift % 1
        plt.figure()
        plt.ylim(0,1)
        plt.xlim(-0.01, 0.51)
        plt.plot(self._k_points, [(x + shift) % 1 for x in self._gaps], 'bD')
        # add plots with +/- 1 to ensure periodicity
        plt.plot(self._k_points, [(x + shift) % 1 + 1 for x in self._gaps], 'bD')
        plt.plot(self._k_points, [(x + shift) % 1 - 1 for x in self._gaps], 'bD')
        for i, kpt in enumerate(self._k_points):
            plt.plot([kpt] * len(self._wcc_list[i]), [(x + shift) % 1 for x in self._wcc_list[i]], "ro")
            # add plots with +/- 1 to ensure periodicity
            plt.plot([kpt] * len(self._wcc_list[i]), [(x + shift) % 1 + 1 for x in self._wcc_list[i]], "ro")
            plt.plot([kpt] * len(self._wcc_list[i]), [(x + shift) % 1 - 1 for x in self._wcc_list[i]], "ro")
        plt.show()
        
    
    def invariant(self):
        """
        calculate the Z2 topological invariant
        """
            
        inv = 1
        for i in range(0, len(self._wcc_list)-1):
            for j in range(0, len(self._wcc_list[0])):
                inv *= self._sgng(self._gaps[i], self._gaps[i+1], self._wcc_list[i+1][j])
        
        return 1 if inv == -1 else 0
        
    #-------------------------------------------------------------------#
    #                support functions for invariants                   #
    #-------------------------------------------------------------------#
    def _sgng(self, z, zplus, x):
        """
        calculates the invariant between two WCC strings
        """
        return np.copysign(1,np.sin(2*np.pi*(zplus - z)) + np.sin(2*np.pi*(x-zplus)) + np.sin(2*np.pi*(z-x)))
    #----------------END SUPPORT FUNCTIONS FOR INVARIANTS---------------#
    
        
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
#                       ABINIT SPECIALISATION                           #
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
class Abinit(Z2PackSystem):
    """
    Abinit Class
    ~~~~~~~~~~~~
    Subclass of Z2PackSystem used for calculating systems using ABINIT
    
    methods: scf, inherits plane
    
    """
    
    def __init__(   self, 
                    name,
                    common_vars_path,
                    psps_files,
                    working_folder,
                    num_occupied,
                    abinit_command = "abinit",
                    nscf_args = {},
                    **kwargs
                    ):
        """
        args:
        ~~~~
        name:
        common_vars_path:       path to ABINIT input file with variables
                                for SCF and NSCF (common variables)
        psps_files:             path to pseudopotential file or list
                                of paths to pseudopotential files
                                MUST BE in the SAME ORDER as the atom
                                types in the common input file
        working_folder:         path to 'build' folder
        num_occupied:           number of occupied bands
        
        kwargs:
        ~~~~~~
        abinit_command:         command to call abinit
        nscf_args:              passed to AbinitRun.nscf
        other kwargs:           are passed to the Z2PackPlane 
                                constructor via .plane(), which passes 
                                them to wcc_calc()
                                precedence: wcc_calc > plane > this
                                (newer kwargs take precedence)
        """
        self._defaults = kwargs
        self._name = name
        self._abinit_system = ar.AbinitRun( name, 
                                            io.parse_input(common_vars_path) , 
                                            psps_files, 
                                            working_folder, 
                                            num_occupied,
                                            abinit_command = abinit_command)
        def _M_handle_creator_abinit(string_dir, plane_pos_dir, plane_pos):
            # check if kx is before or after plane_pos_dir
            if(3 - string_dir > 2 * plane_pos_dir):
                return lambda kx, N: self._abinit_system.nscf(string_dir, [plane_pos, kx], N, **nscf_args)
            else:
                return lambda kx, N: self._abinit_system.nscf(string_dir, [kx, plane_pos], N, **nscf_args)
        self._M_handle_creator = _M_handle_creator_abinit
        
    def scf(self, scf_vars_path, verbose = True, **kwargs):
        """
        args:
        ~~~~
        scf_vars_path:          path to SCF - specific input file
        verbose:                toggles printing
        
        kwargs:
        ~~~~~~
        passed to AbinitRun.scf()
        abinit_args:            ABINIT variables (as dict), will be 
                                added to SCF input file
        
            passed on further to _abinit_run()
            setup_only:         suppresses call to ABINIT, only 
                                creates input files
            clean_subfolder:    toggles deletion of old data when 
                                starting a new calculation 
                                default: True
        """
        self._verbose = verbose
        try:
            if not(kwargs['setup_only']): 
                if(self._verbose):
                    print("starting SCF calculation for " + self._name)
            else:
                if(self._verbose):
                    print("setting up SCF calculation for " + self._name)
        except:
            if(self._verbose):
                print("starting SCF calculation for " + self._name)
        
        self._abinit_system.scf(io.parse_input(scf_vars_path), **kwargs)
        if(self._verbose):
            print("")
        
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
#                    TIGHT BINDING SPECIALISATION                       #
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#

class TightBinding(Z2PackSystem):
    """
    TightBinding Class
    ~~~~~~~~~~~~~~~~~~
    Subclass of Z2PackSystem used for calculating system with a tight-
    binding model
    
    method: inherits plane
    """
    def __init__(   self,
                    tbsystem,
                    **kwargs
                    ):
        """
        args:
        ~~~~
        tbsystem:               TbSystem object
        
        kwargs:
        ~~~~~~
        no TightBinding - specific kwargs
        other kwargs:           are passed to the Z2PackPlane 
                                constructor via .plane(), which passes 
                                them to wcc_calc()
                                precedence: wcc_calc > plane > this
                                (newer kwargs take precedence)
        """
        self._defaults = kwargs
        self._tbsystem = tbsystem
        def _M_handle_creator_tb(string_dir, plane_pos_dir, plane_pos):
            # check if kx is before or after plane_pos_dir
            if(3 - string_dir > 2 * plane_pos_dir):
                return lambda kx, N: self._tbsystem._getM(string_dir, [plane_pos, kx], N)
            else:
                return lambda kx, N: self._tbsystem._getM(string_dir, [kx, plane_pos], N)
        self._M_handle_creator = _M_handle_creator_tb
        
    
    
    

if __name__ == "__main__":
    print("z2pack.py")
