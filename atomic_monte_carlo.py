from shmem4py import shmem
import math
import random

my_rank = shmem.my_pe()
commsz = shmem.n_pes()

def circle(x):
    return 1 - math.pow(x,2)


def approxpi(num):
    count = 0

    for i in range(num):
        x = random.random()
        y = random.random()
        f_x = circle(x)
        if math.pow(y, 2) <= f_x:
            count += 1

    # empty array of size 1 64-bit float (double)
    total = shmem.empty(1, dtype='i')

    # atomic add all the counts together to PE0's total value
    shmem.atomic_add(total, count, 0)
    
    # make sure that all PEs have completed the atomic_add
    # before making pi approximation calculation
    shmem.barrier_all()
    
    # pi = 4 * (points inside / total points)
    return 4 * (float(total[0])/(float(num) * commsz))

appx_num = 10000
appx_pi = approxpi(appx_num)

# only rank 0 stores the total value, so all other PEs would return 0
if my_rank == 0:
    print(f'Pi approximation with {appx_num} events: {appx_pi}')