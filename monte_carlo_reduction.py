from shmem4py import shmem
import math
import random

my_rank = shmem.my_pe()
commsz = shmem.n_pes()

def circle(x):
    return 1 - math.pow(x,2)


def approxpi(num):
    inside = shmem.zeros(1, dtype = 'i')

    for i in range(num):
        x = random.random()
        y = random.random()
        f_x = circle(x)
        if math.pow(y, 2) <= f_x:
            inside[0] += 1

    # make sure PEs are done computing their 'inside' count
    shmem.barrier_all()

    # reduce all PE's inside count with inside count of all other PEs
    # as opposed to atomics, every PE will have the same inside count now
    shmem.sum_reduce(inside, inside)
    
    # pi = 4 * (points inside / total points)
    return 4 * (float(inside[0])/(float(num) * commsz))

appx_num = 100000
appx_pi = approxpi(appx_num)

# only rank 0 stores the total value, so all other PEs would return 0
if my_rank == 0:
    print(f'Pi approximation with {appx_num} events: {appx_pi}')