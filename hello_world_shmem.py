from shmem4py import shmem

# no need to explicitly initialize shmem in shmem4py!

# shmem version of comm sz and comm rank from MPI
# comm size == n_pes()
# comm rank == my_pe()
my_rank = shmem.my_pe()
commsz = shmem.n_pes()

if my_rank == 0:
    print('This is process 0! And I am amazing and I konw that')
else:
    print(f'hi process 0! I am {my_rank} of {commsz} processes')