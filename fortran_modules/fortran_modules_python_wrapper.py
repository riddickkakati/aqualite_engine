'''This Python code is an automatically generated wrapper
for Fortran code made by 'fmodpy'. The original documentation
for the Fortran source code follows.


'''

import os
import ctypes
import platform
import numpy

# --------------------------------------------------------------------
#               CONFIGURATION
# 
_verbose = True
_fort_compiler = "gfortran"
_shared_object_name = "fortran_modules." + platform.machine() + ".so"
_this_directory = os.path.dirname(os.path.abspath(__file__))
_path_to_lib = os.path.join(_this_directory, _shared_object_name)
_compile_options = ['-fPIC', '-shared', '-O3']
_ordered_dependencies = ['fortran_modules.f90', 'fortran_modules_c_wrapper.f90']
_symbol_files = []# 
# --------------------------------------------------------------------
#               AUTO-COMPILING
#
# Try to import the prerequisite symbols for the compiled code.
for _ in _symbol_files:
    _ = ctypes.CDLL(os.path.join(_this_directory, _), mode=ctypes.RTLD_GLOBAL)
# Try to import the existing object. If that fails, recompile and then try.
try:
    # Check to see if the source files have been modified and a recompilation is needed.
    if (max(max([0]+[os.path.getmtime(os.path.realpath(os.path.join(_this_directory,_))) for _ in _symbol_files]),
            max([0]+[os.path.getmtime(os.path.realpath(os.path.join(_this_directory,_))) for _ in _ordered_dependencies]))
        > os.path.getmtime(_path_to_lib)):
        print()
        print("WARNING: Recompiling because the modification time of a source file is newer than the library.", flush=True)
        print()
        if os.path.exists(_path_to_lib):
            os.remove(_path_to_lib)
        raise NotImplementedError(f"The newest library code has not been compiled.")
    # Import the library.
    clib = ctypes.CDLL(_path_to_lib)
except:
    # Remove the shared object if it exists, because it is faulty.
    if os.path.exists(_shared_object_name):
        os.remove(_shared_object_name)
    # Compile a new shared object.
    _command = [_fort_compiler] + _ordered_dependencies + _compile_options + ["-o", _shared_object_name]
    if _verbose:
        print("Running system command with arguments")
        print("  ", " ".join(_command))
    # Run the compilation command.
    import subprocess
    subprocess.check_call(_command, cwd=_this_directory)
    # Import the shared object file as a C library with ctypes.
    clib = ctypes.CDLL(_path_to_lib)
# --------------------------------------------------------------------


# ----------------------------------------------
# Wrapper for the Fortran subroutine LOOP_A2W

def loop_a2w(tw_solution, ta_data, tt, a, tmin, version, dd, pp, k):
    ''''''
    
    # Setting up "tw_solution"
    if (type(tw_solution) is not ctypes.c_double): tw_solution = ctypes.c_double(tw_solution)
    
    # Setting up "ta_data"
    if (type(ta_data) is not ctypes.c_double): ta_data = ctypes.c_double(ta_data)
    
    # Setting up "tt"
    if (type(tt) is not ctypes.c_double): tt = ctypes.c_double(tt)
    
    # Setting up "a"
    if ((not issubclass(type(a), numpy.ndarray)) or
        (not numpy.asarray(a).flags.f_contiguous) or
        (not (a.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'a' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        a = numpy.asarray(a, dtype=ctypes.c_double, order='F')
    a_dim_1 = ctypes.c_long(a.shape[0])
    
    # Setting up "tmin"
    if (type(tmin) is not ctypes.c_double): tmin = ctypes.c_double(tmin)
    
    # Setting up "version"
    if (type(version) is not ctypes.c_int): version = ctypes.c_int(version)
    
    # Setting up "dd"
    if (type(dd) is not ctypes.c_double): dd = ctypes.c_double(dd)
    
    # Setting up "pp"
    if (type(pp) is not ctypes.c_double): pp = ctypes.c_double(pp)
    
    # Setting up "k"
    if (type(k) is not ctypes.c_double): k = ctypes.c_double(k)

    # Call C-accessible Fortran wrapper.
    clib.c_loop_a2w(ctypes.byref(tw_solution), ctypes.byref(ta_data), ctypes.byref(tt), ctypes.byref(a_dim_1), ctypes.c_void_p(a.ctypes.data), ctypes.byref(tmin), ctypes.byref(version), ctypes.byref(dd), ctypes.byref(pp), ctypes.byref(k))

    # Return final results, 'INTENT(OUT)' arguments only.
    return dd.value, pp.value, k.value


# ----------------------------------------------
# Wrapper for the Fortran subroutine SUBSTEP_A2W

def substep_a2w(ta_data1, ta_data, dd, pp, lim, a, dt, dtair, ttt, nsub):
    ''''''
    
    # Setting up "ta_data1"
    if (type(ta_data1) is not ctypes.c_double): ta_data1 = ctypes.c_double(ta_data1)
    
    # Setting up "ta_data"
    if (type(ta_data) is not ctypes.c_double): ta_data = ctypes.c_double(ta_data)
    
    # Setting up "dd"
    if (type(dd) is not ctypes.c_double): dd = ctypes.c_double(dd)
    
    # Setting up "pp"
    if (type(pp) is not ctypes.c_double): pp = ctypes.c_double(pp)
    
    # Setting up "lim"
    if (type(lim) is not ctypes.c_double): lim = ctypes.c_double(lim)
    
    # Setting up "a"
    if ((not issubclass(type(a), numpy.ndarray)) or
        (not numpy.asarray(a).flags.f_contiguous) or
        (not (a.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'a' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        a = numpy.asarray(a, dtype=ctypes.c_double, order='F')
    a_dim_1 = ctypes.c_long(a.shape[0])
    
    # Setting up "dt"
    if (type(dt) is not ctypes.c_double): dt = ctypes.c_double(dt)
    
    # Setting up "dtair"
    if (type(dtair) is not ctypes.c_double): dtair = ctypes.c_double(dtair)
    
    # Setting up "ttt"
    if (type(ttt) is not ctypes.c_double): ttt = ctypes.c_double(ttt)
    
    # Setting up "nsub"
    if (type(nsub) is not ctypes.c_int): nsub = ctypes.c_int(nsub)

    # Call C-accessible Fortran wrapper.
    clib.c_substep_a2w(ctypes.byref(ta_data1), ctypes.byref(ta_data), ctypes.byref(dd), ctypes.byref(pp), ctypes.byref(lim), ctypes.byref(a_dim_1), ctypes.c_void_p(a.ctypes.data), ctypes.byref(dt), ctypes.byref(dtair), ctypes.byref(ttt), ctypes.byref(nsub))

    # Return final results, 'INTENT(OUT)' arguments only.
    return dd.value, pp.value, dt.value, dtair.value, ttt.value, nsub.value


# ----------------------------------------------
# Wrapper for the Fortran subroutine RK4_AIR2STREAM

def rk4_air2stream(tw, ta, q, qmedia, time, a, version, dd, k):
    ''''''
    
    # Setting up "tw"
    if (type(tw) is not ctypes.c_double): tw = ctypes.c_double(tw)
    
    # Setting up "ta"
    if (type(ta) is not ctypes.c_double): ta = ctypes.c_double(ta)
    
    # Setting up "q"
    if (type(q) is not ctypes.c_double): q = ctypes.c_double(q)
    
    # Setting up "qmedia"
    if (type(qmedia) is not ctypes.c_double): qmedia = ctypes.c_double(qmedia)
    
    # Setting up "time"
    if (type(time) is not ctypes.c_double): time = ctypes.c_double(time)
    
    # Setting up "a"
    if ((not issubclass(type(a), numpy.ndarray)) or
        (not numpy.asarray(a).flags.f_contiguous) or
        (not (a.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'a' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        a = numpy.asarray(a, dtype=ctypes.c_double, order='F')
    a_dim_1 = ctypes.c_long(a.shape[0])
    
    # Setting up "version"
    if (type(version) is not ctypes.c_int): version = ctypes.c_int(version)
    
    # Setting up "dd"
    if (type(dd) is not ctypes.c_double): dd = ctypes.c_double(dd)
    
    # Setting up "k"
    if (type(k) is not ctypes.c_double): k = ctypes.c_double(k)

    # Call C-accessible Fortran wrapper.
    clib.c_rk4_air2stream(ctypes.byref(tw), ctypes.byref(ta), ctypes.byref(q), ctypes.byref(qmedia), ctypes.byref(time), ctypes.byref(a_dim_1), ctypes.c_void_p(a.ctypes.data), ctypes.byref(version), ctypes.byref(dd), ctypes.byref(k))

    # Return final results, 'INTENT(OUT)' arguments only.
    return dd.value, k.value


# ----------------------------------------------
# Wrapper for the Fortran subroutine AIR2WATERCN_LOOP

def air2watercn_loop(tw_solution, ta_data, version, tt, a, tmin, nt, dt):
    ''''''
    
    # Setting up "tw_solution"
    if ((not issubclass(type(tw_solution), numpy.ndarray)) or
        (not numpy.asarray(tw_solution).flags.f_contiguous) or
        (not (tw_solution.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'tw_solution' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        tw_solution = numpy.asarray(tw_solution, dtype=ctypes.c_double, order='F')
    tw_solution_dim_1 = ctypes.c_long(tw_solution.shape[0])
    
    # Setting up "ta_data"
    if ((not issubclass(type(ta_data), numpy.ndarray)) or
        (not numpy.asarray(ta_data).flags.f_contiguous) or
        (not (ta_data.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'ta_data' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        ta_data = numpy.asarray(ta_data, dtype=ctypes.c_double, order='F')
    ta_data_dim_1 = ctypes.c_long(ta_data.shape[0])
    
    # Setting up "version"
    if (type(version) is not ctypes.c_int): version = ctypes.c_int(version)
    
    # Setting up "tt"
    if ((not issubclass(type(tt), numpy.ndarray)) or
        (not numpy.asarray(tt).flags.f_contiguous) or
        (not (tt.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'tt' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        tt = numpy.asarray(tt, dtype=ctypes.c_double, order='F')
    tt_dim_1 = ctypes.c_long(tt.shape[0])
    
    # Setting up "a"
    if ((not issubclass(type(a), numpy.ndarray)) or
        (not numpy.asarray(a).flags.f_contiguous) or
        (not (a.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'a' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        a = numpy.asarray(a, dtype=ctypes.c_double, order='F')
    a_dim_1 = ctypes.c_long(a.shape[0])
    
    # Setting up "tmin"
    if (type(tmin) is not ctypes.c_double): tmin = ctypes.c_double(tmin)
    
    # Setting up "nt"
    if (type(nt) is not ctypes.c_int): nt = ctypes.c_int(nt)
    
    # Setting up "dt"
    if (type(dt) is not ctypes.c_double): dt = ctypes.c_double(dt)

    # Call C-accessible Fortran wrapper.
    clib.c_air2watercn_loop(ctypes.byref(tw_solution_dim_1), ctypes.c_void_p(tw_solution.ctypes.data), ctypes.byref(ta_data_dim_1), ctypes.c_void_p(ta_data.ctypes.data), ctypes.byref(version), ctypes.byref(tt_dim_1), ctypes.c_void_p(tt.ctypes.data), ctypes.byref(a_dim_1), ctypes.c_void_p(a.ctypes.data), ctypes.byref(tmin), ctypes.byref(nt), ctypes.byref(dt))

    # Return final results, 'INTENT(OUT)' arguments only.
    return tw_solution, version.value


# ----------------------------------------------
# Wrapper for the Fortran subroutine AIR2WATEREULER_LOOP

def air2watereuler_loop(tw_solution, ta_data, version, cfl, tt, a, tmin, nt, dt):
    ''''''
    
    # Setting up "tw_solution"
    if ((not issubclass(type(tw_solution), numpy.ndarray)) or
        (not numpy.asarray(tw_solution).flags.f_contiguous) or
        (not (tw_solution.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'tw_solution' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        tw_solution = numpy.asarray(tw_solution, dtype=ctypes.c_double, order='F')
    tw_solution_dim_1 = ctypes.c_long(tw_solution.shape[0])
    
    # Setting up "ta_data"
    if ((not issubclass(type(ta_data), numpy.ndarray)) or
        (not numpy.asarray(ta_data).flags.f_contiguous) or
        (not (ta_data.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'ta_data' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        ta_data = numpy.asarray(ta_data, dtype=ctypes.c_double, order='F')
    ta_data_dim_1 = ctypes.c_long(ta_data.shape[0])
    
    # Setting up "version"
    if (type(version) is not ctypes.c_int): version = ctypes.c_int(version)
    
    # Setting up "cfl"
    if (type(cfl) is not ctypes.c_double): cfl = ctypes.c_double(cfl)
    
    # Setting up "tt"
    if ((not issubclass(type(tt), numpy.ndarray)) or
        (not numpy.asarray(tt).flags.f_contiguous) or
        (not (tt.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'tt' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        tt = numpy.asarray(tt, dtype=ctypes.c_double, order='F')
    tt_dim_1 = ctypes.c_long(tt.shape[0])
    
    # Setting up "a"
    if ((not issubclass(type(a), numpy.ndarray)) or
        (not numpy.asarray(a).flags.f_contiguous) or
        (not (a.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'a' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        a = numpy.asarray(a, dtype=ctypes.c_double, order='F')
    a_dim_1 = ctypes.c_long(a.shape[0])
    
    # Setting up "tmin"
    if (type(tmin) is not ctypes.c_double): tmin = ctypes.c_double(tmin)
    
    # Setting up "nt"
    if (type(nt) is not ctypes.c_int): nt = ctypes.c_int(nt)
    
    # Setting up "dt"
    if (type(dt) is not ctypes.c_double): dt = ctypes.c_double(dt)

    # Call C-accessible Fortran wrapper.
    clib.c_air2watereuler_loop(ctypes.byref(tw_solution_dim_1), ctypes.c_void_p(tw_solution.ctypes.data), ctypes.byref(ta_data_dim_1), ctypes.c_void_p(ta_data.ctypes.data), ctypes.byref(version), ctypes.byref(cfl), ctypes.byref(tt_dim_1), ctypes.c_void_p(tt.ctypes.data), ctypes.byref(a_dim_1), ctypes.c_void_p(a.ctypes.data), ctypes.byref(tmin), ctypes.byref(nt), ctypes.byref(dt))

    # Return final results, 'INTENT(OUT)' arguments only.
    return tw_solution, version.value, cfl.value


# ----------------------------------------------
# Wrapper for the Fortran subroutine AIR2WATERRK2_LOOP

def air2waterrk2_loop(tw_solution, ta_data, version, cfl, tt, a, tmin, nt, dt):
    ''''''
    
    # Setting up "tw_solution"
    if ((not issubclass(type(tw_solution), numpy.ndarray)) or
        (not numpy.asarray(tw_solution).flags.f_contiguous) or
        (not (tw_solution.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'tw_solution' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        tw_solution = numpy.asarray(tw_solution, dtype=ctypes.c_double, order='F')
    tw_solution_dim_1 = ctypes.c_long(tw_solution.shape[0])
    
    # Setting up "ta_data"
    if ((not issubclass(type(ta_data), numpy.ndarray)) or
        (not numpy.asarray(ta_data).flags.f_contiguous) or
        (not (ta_data.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'ta_data' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        ta_data = numpy.asarray(ta_data, dtype=ctypes.c_double, order='F')
    ta_data_dim_1 = ctypes.c_long(ta_data.shape[0])
    
    # Setting up "version"
    if (type(version) is not ctypes.c_int): version = ctypes.c_int(version)
    
    # Setting up "cfl"
    if (type(cfl) is not ctypes.c_double): cfl = ctypes.c_double(cfl)
    
    # Setting up "tt"
    if ((not issubclass(type(tt), numpy.ndarray)) or
        (not numpy.asarray(tt).flags.f_contiguous) or
        (not (tt.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'tt' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        tt = numpy.asarray(tt, dtype=ctypes.c_double, order='F')
    tt_dim_1 = ctypes.c_long(tt.shape[0])
    
    # Setting up "a"
    if ((not issubclass(type(a), numpy.ndarray)) or
        (not numpy.asarray(a).flags.f_contiguous) or
        (not (a.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'a' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        a = numpy.asarray(a, dtype=ctypes.c_double, order='F')
    a_dim_1 = ctypes.c_long(a.shape[0])
    
    # Setting up "tmin"
    if (type(tmin) is not ctypes.c_double): tmin = ctypes.c_double(tmin)
    
    # Setting up "nt"
    if (type(nt) is not ctypes.c_int): nt = ctypes.c_int(nt)
    
    # Setting up "dt"
    if (type(dt) is not ctypes.c_double): dt = ctypes.c_double(dt)

    # Call C-accessible Fortran wrapper.
    clib.c_air2waterrk2_loop(ctypes.byref(tw_solution_dim_1), ctypes.c_void_p(tw_solution.ctypes.data), ctypes.byref(ta_data_dim_1), ctypes.c_void_p(ta_data.ctypes.data), ctypes.byref(version), ctypes.byref(cfl), ctypes.byref(tt_dim_1), ctypes.c_void_p(tt.ctypes.data), ctypes.byref(a_dim_1), ctypes.c_void_p(a.ctypes.data), ctypes.byref(tmin), ctypes.byref(nt), ctypes.byref(dt))

    # Return final results, 'INTENT(OUT)' arguments only.
    return tw_solution, version.value, cfl.value


# ----------------------------------------------
# Wrapper for the Fortran subroutine AIR2WATERRK4_LOOP

def air2waterrk4_loop(tw_solution, ta_data, version, cfl, tt, a, tmin, nt, dt):
    ''''''
    
    # Setting up "tw_solution"
    if ((not issubclass(type(tw_solution), numpy.ndarray)) or
        (not numpy.asarray(tw_solution).flags.f_contiguous) or
        (not (tw_solution.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'tw_solution' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        tw_solution = numpy.asarray(tw_solution, dtype=ctypes.c_double, order='F')
    tw_solution_dim_1 = ctypes.c_long(tw_solution.shape[0])
    
    # Setting up "ta_data"
    if ((not issubclass(type(ta_data), numpy.ndarray)) or
        (not numpy.asarray(ta_data).flags.f_contiguous) or
        (not (ta_data.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'ta_data' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        ta_data = numpy.asarray(ta_data, dtype=ctypes.c_double, order='F')
    ta_data_dim_1 = ctypes.c_long(ta_data.shape[0])
    
    # Setting up "version"
    if (type(version) is not ctypes.c_int): version = ctypes.c_int(version)
    
    # Setting up "cfl"
    if (type(cfl) is not ctypes.c_double): cfl = ctypes.c_double(cfl)
    
    # Setting up "tt"
    if ((not issubclass(type(tt), numpy.ndarray)) or
        (not numpy.asarray(tt).flags.f_contiguous) or
        (not (tt.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'tt' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        tt = numpy.asarray(tt, dtype=ctypes.c_double, order='F')
    tt_dim_1 = ctypes.c_long(tt.shape[0])
    
    # Setting up "a"
    if ((not issubclass(type(a), numpy.ndarray)) or
        (not numpy.asarray(a).flags.f_contiguous) or
        (not (a.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'a' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        a = numpy.asarray(a, dtype=ctypes.c_double, order='F')
    a_dim_1 = ctypes.c_long(a.shape[0])
    
    # Setting up "tmin"
    if (type(tmin) is not ctypes.c_double): tmin = ctypes.c_double(tmin)
    
    # Setting up "nt"
    if (type(nt) is not ctypes.c_int): nt = ctypes.c_int(nt)
    
    # Setting up "dt"
    if (type(dt) is not ctypes.c_double): dt = ctypes.c_double(dt)

    # Call C-accessible Fortran wrapper.
    clib.c_air2waterrk4_loop(ctypes.byref(tw_solution_dim_1), ctypes.c_void_p(tw_solution.ctypes.data), ctypes.byref(ta_data_dim_1), ctypes.c_void_p(ta_data.ctypes.data), ctypes.byref(version), ctypes.byref(cfl), ctypes.byref(tt_dim_1), ctypes.c_void_p(tt.ctypes.data), ctypes.byref(a_dim_1), ctypes.c_void_p(a.ctypes.data), ctypes.byref(tmin), ctypes.byref(nt), ctypes.byref(dt))

    # Return final results, 'INTENT(OUT)' arguments only.
    return tw_solution, version.value, cfl.value


# ----------------------------------------------
# Wrapper for the Fortran subroutine AIR2STREAMCN_LOOP

def air2streamcn_loop(tw_solution, ta_data, q, qmedia, version, tt, a, tmin, nt, dt):
    ''''''
    
    # Setting up "tw_solution"
    if ((not issubclass(type(tw_solution), numpy.ndarray)) or
        (not numpy.asarray(tw_solution).flags.f_contiguous) or
        (not (tw_solution.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'tw_solution' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        tw_solution = numpy.asarray(tw_solution, dtype=ctypes.c_double, order='F')
    tw_solution_dim_1 = ctypes.c_long(tw_solution.shape[0])
    
    # Setting up "ta_data"
    if ((not issubclass(type(ta_data), numpy.ndarray)) or
        (not numpy.asarray(ta_data).flags.f_contiguous) or
        (not (ta_data.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'ta_data' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        ta_data = numpy.asarray(ta_data, dtype=ctypes.c_double, order='F')
    ta_data_dim_1 = ctypes.c_long(ta_data.shape[0])
    
    # Setting up "q"
    if ((not issubclass(type(q), numpy.ndarray)) or
        (not numpy.asarray(q).flags.f_contiguous) or
        (not (q.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'q' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        q = numpy.asarray(q, dtype=ctypes.c_double, order='F')
    q_dim_1 = ctypes.c_long(q.shape[0])
    
    # Setting up "qmedia"
    if (type(qmedia) is not ctypes.c_double): qmedia = ctypes.c_double(qmedia)
    
    # Setting up "version"
    if (type(version) is not ctypes.c_double): version = ctypes.c_double(version)
    
    # Setting up "tt"
    if ((not issubclass(type(tt), numpy.ndarray)) or
        (not numpy.asarray(tt).flags.f_contiguous) or
        (not (tt.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'tt' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        tt = numpy.asarray(tt, dtype=ctypes.c_double, order='F')
    tt_dim_1 = ctypes.c_long(tt.shape[0])
    
    # Setting up "a"
    if ((not issubclass(type(a), numpy.ndarray)) or
        (not numpy.asarray(a).flags.f_contiguous) or
        (not (a.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'a' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        a = numpy.asarray(a, dtype=ctypes.c_double, order='F')
    a_dim_1 = ctypes.c_long(a.shape[0])
    
    # Setting up "tmin"
    if (type(tmin) is not ctypes.c_double): tmin = ctypes.c_double(tmin)
    
    # Setting up "nt"
    if (type(nt) is not ctypes.c_int): nt = ctypes.c_int(nt)
    
    # Setting up "dt"
    if (type(dt) is not ctypes.c_double): dt = ctypes.c_double(dt)

    # Call C-accessible Fortran wrapper.
    clib.c_air2streamcn_loop(ctypes.byref(tw_solution_dim_1), ctypes.c_void_p(tw_solution.ctypes.data), ctypes.byref(ta_data_dim_1), ctypes.c_void_p(ta_data.ctypes.data), ctypes.byref(q_dim_1), ctypes.c_void_p(q.ctypes.data), ctypes.byref(qmedia), ctypes.byref(version), ctypes.byref(tt_dim_1), ctypes.c_void_p(tt.ctypes.data), ctypes.byref(a_dim_1), ctypes.c_void_p(a.ctypes.data), ctypes.byref(tmin), ctypes.byref(nt), ctypes.byref(dt))

    # Return final results, 'INTENT(OUT)' arguments only.
    return tw_solution


# ----------------------------------------------
# Wrapper for the Fortran subroutine AIR2STREAMEULER_LOOP

def air2streameuler_loop(tw_solution, ta_data, q, qmedia, version, tt, a, tmin, nt, dt):
    ''''''
    
    # Setting up "tw_solution"
    if ((not issubclass(type(tw_solution), numpy.ndarray)) or
        (not numpy.asarray(tw_solution).flags.f_contiguous) or
        (not (tw_solution.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'tw_solution' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        tw_solution = numpy.asarray(tw_solution, dtype=ctypes.c_double, order='F')
    tw_solution_dim_1 = ctypes.c_long(tw_solution.shape[0])
    
    # Setting up "ta_data"
    if ((not issubclass(type(ta_data), numpy.ndarray)) or
        (not numpy.asarray(ta_data).flags.f_contiguous) or
        (not (ta_data.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'ta_data' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        ta_data = numpy.asarray(ta_data, dtype=ctypes.c_double, order='F')
    ta_data_dim_1 = ctypes.c_long(ta_data.shape[0])
    
    # Setting up "q"
    if ((not issubclass(type(q), numpy.ndarray)) or
        (not numpy.asarray(q).flags.f_contiguous) or
        (not (q.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'q' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        q = numpy.asarray(q, dtype=ctypes.c_double, order='F')
    q_dim_1 = ctypes.c_long(q.shape[0])
    
    # Setting up "qmedia"
    if (type(qmedia) is not ctypes.c_double): qmedia = ctypes.c_double(qmedia)
    
    # Setting up "version"
    if (type(version) is not ctypes.c_int): version = ctypes.c_int(version)
    
    # Setting up "tt"
    if ((not issubclass(type(tt), numpy.ndarray)) or
        (not numpy.asarray(tt).flags.f_contiguous) or
        (not (tt.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'tt' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        tt = numpy.asarray(tt, dtype=ctypes.c_double, order='F')
    tt_dim_1 = ctypes.c_long(tt.shape[0])
    
    # Setting up "a"
    if ((not issubclass(type(a), numpy.ndarray)) or
        (not numpy.asarray(a).flags.f_contiguous) or
        (not (a.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'a' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        a = numpy.asarray(a, dtype=ctypes.c_double, order='F')
    a_dim_1 = ctypes.c_long(a.shape[0])
    
    # Setting up "tmin"
    if (type(tmin) is not ctypes.c_double): tmin = ctypes.c_double(tmin)
    
    # Setting up "nt"
    if (type(nt) is not ctypes.c_int): nt = ctypes.c_int(nt)
    
    # Setting up "dt"
    if (type(dt) is not ctypes.c_double): dt = ctypes.c_double(dt)

    # Call C-accessible Fortran wrapper.
    clib.c_air2streameuler_loop(ctypes.byref(tw_solution_dim_1), ctypes.c_void_p(tw_solution.ctypes.data), ctypes.byref(ta_data_dim_1), ctypes.c_void_p(ta_data.ctypes.data), ctypes.byref(q_dim_1), ctypes.c_void_p(q.ctypes.data), ctypes.byref(qmedia), ctypes.byref(version), ctypes.byref(tt_dim_1), ctypes.c_void_p(tt.ctypes.data), ctypes.byref(a_dim_1), ctypes.c_void_p(a.ctypes.data), ctypes.byref(tmin), ctypes.byref(nt), ctypes.byref(dt))

    # Return final results, 'INTENT(OUT)' arguments only.
    return tw_solution


# ----------------------------------------------
# Wrapper for the Fortran subroutine AIR2STREAMRK2_LOOP

def air2streamrk2_loop(tw_solution, ta_data, q, qmedia, version, tt, a, tmin, nt, dt):
    ''''''
    
    # Setting up "tw_solution"
    if ((not issubclass(type(tw_solution), numpy.ndarray)) or
        (not numpy.asarray(tw_solution).flags.f_contiguous) or
        (not (tw_solution.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'tw_solution' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        tw_solution = numpy.asarray(tw_solution, dtype=ctypes.c_double, order='F')
    tw_solution_dim_1 = ctypes.c_long(tw_solution.shape[0])
    
    # Setting up "ta_data"
    if ((not issubclass(type(ta_data), numpy.ndarray)) or
        (not numpy.asarray(ta_data).flags.f_contiguous) or
        (not (ta_data.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'ta_data' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        ta_data = numpy.asarray(ta_data, dtype=ctypes.c_double, order='F')
    ta_data_dim_1 = ctypes.c_long(ta_data.shape[0])
    
    # Setting up "q"
    if ((not issubclass(type(q), numpy.ndarray)) or
        (not numpy.asarray(q).flags.f_contiguous) or
        (not (q.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'q' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        q = numpy.asarray(q, dtype=ctypes.c_double, order='F')
    q_dim_1 = ctypes.c_long(q.shape[0])
    
    # Setting up "qmedia"
    if (type(qmedia) is not ctypes.c_double): qmedia = ctypes.c_double(qmedia)
    
    # Setting up "version"
    if (type(version) is not ctypes.c_int): version = ctypes.c_int(version)
    
    # Setting up "tt"
    if ((not issubclass(type(tt), numpy.ndarray)) or
        (not numpy.asarray(tt).flags.f_contiguous) or
        (not (tt.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'tt' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        tt = numpy.asarray(tt, dtype=ctypes.c_double, order='F')
    tt_dim_1 = ctypes.c_long(tt.shape[0])
    
    # Setting up "a"
    if ((not issubclass(type(a), numpy.ndarray)) or
        (not numpy.asarray(a).flags.f_contiguous) or
        (not (a.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'a' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        a = numpy.asarray(a, dtype=ctypes.c_double, order='F')
    a_dim_1 = ctypes.c_long(a.shape[0])
    
    # Setting up "tmin"
    if (type(tmin) is not ctypes.c_double): tmin = ctypes.c_double(tmin)
    
    # Setting up "nt"
    if (type(nt) is not ctypes.c_int): nt = ctypes.c_int(nt)
    
    # Setting up "dt"
    if (type(dt) is not ctypes.c_double): dt = ctypes.c_double(dt)

    # Call C-accessible Fortran wrapper.
    clib.c_air2streamrk2_loop(ctypes.byref(tw_solution_dim_1), ctypes.c_void_p(tw_solution.ctypes.data), ctypes.byref(ta_data_dim_1), ctypes.c_void_p(ta_data.ctypes.data), ctypes.byref(q_dim_1), ctypes.c_void_p(q.ctypes.data), ctypes.byref(qmedia), ctypes.byref(version), ctypes.byref(tt_dim_1), ctypes.c_void_p(tt.ctypes.data), ctypes.byref(a_dim_1), ctypes.c_void_p(a.ctypes.data), ctypes.byref(tmin), ctypes.byref(nt), ctypes.byref(dt))

    # Return final results, 'INTENT(OUT)' arguments only.
    return tw_solution


# ----------------------------------------------
# Wrapper for the Fortran subroutine AIR2STREAMRK4_LOOP

def air2streamrk4_loop(tw_solution, ta_data, q, qmedia, version, tt, a, tmin, nt, dt):
    ''''''
    
    # Setting up "tw_solution"
    if ((not issubclass(type(tw_solution), numpy.ndarray)) or
        (not numpy.asarray(tw_solution).flags.f_contiguous) or
        (not (tw_solution.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'tw_solution' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        tw_solution = numpy.asarray(tw_solution, dtype=ctypes.c_double, order='F')
    tw_solution_dim_1 = ctypes.c_long(tw_solution.shape[0])
    
    # Setting up "ta_data"
    if ((not issubclass(type(ta_data), numpy.ndarray)) or
        (not numpy.asarray(ta_data).flags.f_contiguous) or
        (not (ta_data.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'ta_data' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        ta_data = numpy.asarray(ta_data, dtype=ctypes.c_double, order='F')
    ta_data_dim_1 = ctypes.c_long(ta_data.shape[0])
    
    # Setting up "q"
    if ((not issubclass(type(q), numpy.ndarray)) or
        (not numpy.asarray(q).flags.f_contiguous) or
        (not (q.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'q' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        q = numpy.asarray(q, dtype=ctypes.c_double, order='F')
    q_dim_1 = ctypes.c_long(q.shape[0])
    
    # Setting up "qmedia"
    if (type(qmedia) is not ctypes.c_double): qmedia = ctypes.c_double(qmedia)
    
    # Setting up "version"
    if (type(version) is not ctypes.c_int): version = ctypes.c_int(version)
    
    # Setting up "tt"
    if ((not issubclass(type(tt), numpy.ndarray)) or
        (not numpy.asarray(tt).flags.f_contiguous) or
        (not (tt.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'tt' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        tt = numpy.asarray(tt, dtype=ctypes.c_double, order='F')
    tt_dim_1 = ctypes.c_long(tt.shape[0])
    
    # Setting up "a"
    if ((not issubclass(type(a), numpy.ndarray)) or
        (not numpy.asarray(a).flags.f_contiguous) or
        (not (a.dtype == numpy.dtype(ctypes.c_double)))):
        import warnings
        warnings.warn("The provided argument 'a' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
        a = numpy.asarray(a, dtype=ctypes.c_double, order='F')
    a_dim_1 = ctypes.c_long(a.shape[0])
    
    # Setting up "tmin"
    if (type(tmin) is not ctypes.c_double): tmin = ctypes.c_double(tmin)
    
    # Setting up "nt"
    if (type(nt) is not ctypes.c_int): nt = ctypes.c_int(nt)
    
    # Setting up "dt"
    if (type(dt) is not ctypes.c_double): dt = ctypes.c_double(dt)

    # Call C-accessible Fortran wrapper.
    clib.c_air2streamrk4_loop(ctypes.byref(tw_solution_dim_1), ctypes.c_void_p(tw_solution.ctypes.data), ctypes.byref(ta_data_dim_1), ctypes.c_void_p(ta_data.ctypes.data), ctypes.byref(q_dim_1), ctypes.c_void_p(q.ctypes.data), ctypes.byref(qmedia), ctypes.byref(version), ctypes.byref(tt_dim_1), ctypes.c_void_p(tt.ctypes.data), ctypes.byref(a_dim_1), ctypes.c_void_p(a.ctypes.data), ctypes.byref(tmin), ctypes.byref(nt), ctypes.byref(dt))

    # Return final results, 'INTENT(OUT)' arguments only.
    return tw_solution

