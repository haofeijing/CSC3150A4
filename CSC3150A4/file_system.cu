#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__device__ __managed__ u32 gtime = 0;


__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
							int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
							int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE, 
							int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS)
{
  // init variables
  fs->volume = volume;

  // init constants
  fs->SUPERBLOCK_SIZE = SUPERBLOCK_SIZE;
  fs->FCB_SIZE = FCB_SIZE;
  fs->FCB_ENTRIES = FCB_ENTRIES;
  fs->STORAGE_SIZE = VOLUME_SIZE;
  fs->STORAGE_BLOCK_SIZE = STORAGE_BLOCK_SIZE;
  fs->MAX_FILENAME_SIZE = MAX_FILENAME_SIZE;
  fs->MAX_FILE_NUM = MAX_FILE_NUM;
  fs->MAX_FILE_SIZE = MAX_FILE_SIZE;
  fs->FILE_BASE_ADDRESS = FILE_BASE_ADDRESS;

}



__device__ u32 fs_open(FileSystem *fs, char *s, int op)
{
	/* Implement open operation here */

	// get file name
	//int i = 0;
	//char tmp = *s;
	//while (tmp != '\0') {
	//	printf("%c", tmp);
	//	i++;
	//	tmp = *(s + i);
	//}
	//printf("\n");

	// find existed file
	int i = fs->SUPERBLOCK_SIZE;
	bool find = false;
	int empty_entry = -1;
	char tmp;
	int j;
	int idx;
	bool flag;
	while (i < fs->SUPERBLOCK_SIZE + fs->FCB_ENTRIES * fs->FCB_SIZE) {
		idx = (i - fs->SUPERBLOCK_SIZE) / fs->FCB_SIZE;
		if (fs->volume[idx] != 1) {
			// record empty entry;
			if (empty_entry == -1) {
				emtpry_entry = idx;
			} 
			i += fs->FCB_SIZE;
			continue; // skip empty entry
		}
		tmp = *s;
		j = 0;
		flag = true;
		while (tmp != '\0') {
			if (fs->volume[i + j] != tmp) {
				flag = false;
				break;
			}
			else
			{
				j++;
				tmp = *(s + j);
			}
		}
		if (!flag) {
			printf("this block = %d\n", i);
			printf("not find in this block\n");
			i += fs->FCB_SIZE;
			continue;
		}
		else
		{
			find = true;
			// return pointer, i.e. current entry since we have 1024 blocks for 1024 files.
			// We allocate 32 blocks for each file.
			return fs->FILE_BASE_ADDRESS + idx * (fs->STORAGE_BLOCK_SIZE * 32); 
			
		}
	}
	if (!find) {
		if (op == G_READ) {
			printf("ERROR: no such file.\n");
			return 0;
		} else if (op == G_WRITE) {
			fs->volume[empty] = 1; // change status of FCB in super block
			// write file name in FCB
			tmp = *s;
			j = 0;
			while (tmp != '\0') {
				fs->volume[fs->SUPERBLOCK_SIZE + empty * fs->FCB_SIZE + j] = tmp;
				j++;
				tmp = *(s + j);
			}
			return fs->FILE_BASE_ADDRESS + empty * (fs->STORAGE_BLOCK_SIZE * 32);
			

		} else {
			printf("ERROR: invalid operation.\n");
			return 0;
		}
		

	}
}


__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp)
{
	/* Implement read operation here */
}

__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp)
{
	/* Implement write operation here */
	

}
__device__ void fs_gsys(FileSystem *fs, int op)
{
	/* Implement LS_D and LS_S operation here */
}

__device__ void fs_gsys(FileSystem *fs, int op, char *s)
{
	/* Implement rm operation here */
}
