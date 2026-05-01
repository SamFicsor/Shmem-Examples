from shmem4py import shmem
import numpy as np

my_rank = shmem.my_pe()
commsz = shmem.n_pes()

# PART 2: BROADCASTING AN ARRAY

# allocates a new NumPy array of 0s in the *symmetric memory*
# size of array = commsz
# data type must match, so we specify as 32 bit integer
source_arr = shmem.zeros(commsz, dtype="int32")
print(f'Array at rank {my_rank} = {source_arr}')

# allocates a new NumPy array filled with the specified value = -999
# same as above, but allowed to fill with other values besides 0
# STILL IN SYMMETRIC MEMORY
dest_arr = shmem.full(commsz, -999, dtype="int32")

# rank 0 edits the source array
if my_rank == 0:
    for i in range(commsz):
        source_arr[i] = i + 1
    print(f'Rank 0 edits: {source_arr}')

# barrier so that all the PEs get the same final array
shmem.barrier_all()

# all PEs get PE0's version of source_arr broadcasted to the local dest_arr
shmem.broadcast(dest_arr, source_arr, 0)

print(f'Array at rank {my_rank} after broadcast: {dest_arr}')
print()

shmem.free(source_arr)
shmem.free(dest_arr)