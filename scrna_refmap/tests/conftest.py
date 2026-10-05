"""Limit BLAS/OpenMP threads: the test datasets are small and oversubscription on
shared machines slows them by an order of magnitude."""
import pytest
from threadpoolctl import threadpool_limits


@pytest.fixture(scope="session", autouse=True)
def _single_thread_blas():
    with threadpool_limits(1):
        yield
