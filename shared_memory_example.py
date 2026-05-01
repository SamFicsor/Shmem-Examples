from shmem4py import shmem
import numpy as np

my_rank = shmem.my_pe()
commsz = shmem.n_pes()

# PART 1: GET A SINGLE ITEM

# all PEs must have 
nextpe = (my_rank + 1) % commsz

# our symmetric memory is stored using np arrays, as these are mutable
# we need to specify the size and data type of the array like we would in most other languages
# this allows the correct amount of memory to be allocated
# so all PEs know where to look in the shared memory (global address)
src = shmem.empty(1, dtype='i') # empty array of size 1 int
src[0] = my_rank

dst = np.empty(1, dtype='i')
dst[0] = -1

print(f'Before grabbing data: target at rank {my_rank} = {dst[0]}')

shmem.barrier_all()

# each PE will get the rank of the next PE and put that into the 'target' variable locally
shmem.get(dst, src, nextpe)

print(f'After grabbing data: target at rank {my_rank} = {dst[0]}')

# src[0] is the target, and it changes after the get operation is executed by each PE

# because this is a wrapper for C-based SHMEM, still need to free the memory allocated
shmem.free(src)
shmem.free(dst)