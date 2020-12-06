"""Implementation of the MPI collective group."""
try:
    import mpi4py
except ImportError:
    raise ImportError("mpi4py fail to import, please run 'conda install mpi4py' to in stall it")

import logging
import datetime
import time

import ray
import numpy as np
import mpi4py.MPI as MPI

from ray.util.collective.collective_group import mpi_util
from ray.util.collective.collective_group.base_collective_group import BaseGroup
from ray.util.collective.types import AllReduceOptions, BarrierOptions
from ray.util.collective.const import NAMED_ACTOR_STORE_SUFFIX

logger = logging.getLogger(__name__)

class Rendezvous:
    def __init__(self, group_name):
        if not group_name:
            raise ValueError('Empty meeting point.')
        self._group_name = group_name
        self._store_name = None
        self._store = None

    def meet_at_store(self, timeout=180):
        """Meet at the named actor store."""
        if timeout is not None and timeout < 0:
            raise ValueError("The 'timeout' argument must be nonnegative. "
                             f"Received {timeout}")
        self._store_name = self._group_name + NAMED_ACTOR_STORE_SUFFIX
        timeout_delta = datetime.timedelta(seconds=timeout)
        elapsed = datetime.timedelta(seconds=0)
        start_time = datetime.datetime.now()
        while elapsed < timeout_delta:
            try:
                logger.debug("Trying to meet at the store '{}'".format(self._store_name))
                self._store = ray.get_actor(self._store_name)
            except ValueError:
                logger.debug("Failed to meet at the store '{}'."
                              "Trying again...".format(self._store_name))
                time.sleep(1)
                elapsed = datetime.datetime.now() - start_time
                continue
            logger.debug("Successful rendezvous!")
            break
        if not self._store:
            raise RuntimeError("Unable to meet other processes "
                               "at the rendezvous store.")

    @property
    def store(self):
        return self._store

    def get_mpi_id(self):
        if not self._store:
            raise ValueError("Rendezvous store is not setup.")
        uid = ray.get(self._store.get_id.remote())
        return uid

class MPIGroup(BaseGroup):
    def __init__(self, world_size, rank, group_name):
        """Init an MPI collective group."""
        super(MPIGroup, self).__init__(world_size, rank, group_name)
        print('rank {} group_name {}'.format(rank, group_name))

        # default communicator
        self._mpi_comm = MPI.COMM_WORLD

        _rank = self._mpi_comm.rank
        print(_rank, rank)
        # assert _rank == rank

        self._rendezvous = Rendezvous(self.group_name)
        self._rendezvous.meet_at_store()

        # Setup the mpi uid using the store
        self._init_mpi_unique_id()

        # # Setup a tensor for barrier calls
        # self._barrier_tensor = np.array([1])

    def _init_mpi_unique_id(self):
        """Init the MPI unique ID required for setting up MPI communicator."""
        self._mpi_uid = self._rendezvous.get_mpi_id()

    @classmethod
    def backend(cls):
        return 'mpi'

    @property
    def rank(self):
        return self._mpi_comm.rank

    @property
    def size(self):
        return self._mpi_comm.size

    def destroy_group(self):
        """Destroy the group and release the MPI communicators safely."""
        MPI.Finalize()

    def allreduce(self, tensor, allreduce_options=AllReduceOptions()):
        """
        AllReduce a list of tensors following options.

        Args:
            tensor: the tensor to be reduced, each tensor locates on a GPU
            allreduce_options:

        Returns:
        """
        mpi_util._check_dtype("allreduce", tensor)

        dtype = mpi_util.get_mpi_tensor_dtype(tensor)
        op = mpi_util.get_mpi_reduce_op(allreduce_options.reduceOp)

        self._mpi_comm.Allreduce(MPI.IN_PLACE, [tensor, dtype], op=op)

    def barrier(self, barrier_options=BarrierOptions()):
        self._mpi_comm.Barrier()
