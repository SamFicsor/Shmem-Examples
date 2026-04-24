//HelloWorld.c

#include <stdio.h>
#include <shmem.h>

int main(){
	shmem_init();

	int my_pe, num_pe;         //declare variables for both pe id of processor and the number of pes
	

	num_pe = shmem_n_pes();    //obtain the number of pes that can be used
	my_pe  = shmem_my_pe();    //obtain the pe id number
	
	printf("Hello from %d of %d\n", my_pe, num_pe);
	shmem_finalize();
	return 0;
}