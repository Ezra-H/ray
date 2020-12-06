
import ray
import numpy as np

import ray.util.collective as collective

@ray.remote(num_gpus=1)
class Worker:
    def __init__(self):
        self.send = np.ones((4,), dtype=np.float32)
        self.recv = np.zeros((4,), dtype=np.float32)

    def setup(self, world_size, rank):
        collective.init_collective_group('mpi', world_size, rank, 'default')
        return True

    def compute(self):
        collective.allreduce(self.send, 'default')
        print(self.send)
        return self.send

    def destroy(self):
        collective.destroy_group('')

if __name__ == "__main__":

    send = np.ones((4, ), dtype=np.float32)

    ray.init(num_gpus=2)

    num_workers = 2
    workers = []
    init_rets = []
    for i in range(num_workers):
        print(i)
        w = Worker.remote()
        workers.append(w)
        m = ray.get(w.setup.remote(num_workers, i))
        init_rets.append(m)
    results = ray.get([w.compute.remote() for w in workers])
    # print(results)
    ray.shutdown()