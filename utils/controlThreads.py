import os

THREADS = 16


def prepare_env():
    os.environ["OMP_NUM_THREADS"] = str(THREADS)
    os.environ["OPENBLAS_NUM_THREADS"] = str(THREADS)
    os.environ["MKL_NUM_THREADS"] = str(THREADS)
    os.environ["NUMEXPR_NUM_THREADS"] = str(THREADS)
    os.environ["NUMBA_NUM_THREADS"] = str(THREADS)
