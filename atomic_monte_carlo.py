from shmem4py import shmem
import math
import random

def circle(x):
    return 1 - math.pow(x,2)


def approxpi(num):
    my_rank = shmem.my_pe()
    commsz = shmem.n_pes()

    count = 0

    for i in range(num):
        x = random.random()
        y = random.random()
        f_x = circle(x)
        if math.pow(y, 2) <= f_x:
            count += 1

    shmem.atomic_add(target, value, be)

appx_num = 10000
appx_pi = approxpi(appx_num)

print(f'Pi approximation with {appx_num} events: {appx_pi}')