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
				empty_entry = idx;
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
			//printf("this block = %d\n", i);
			//printf("not find in this block\n");
			i += fs->FCB_SIZE;
			continue;
		}
		else
		{
			find = true;
			// change modified time
			int modify_idx = fs->SUPERBLOCK_SIZE + idx * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE;
			for (int k = 0; k < 4; k++) {
				fs->volume[modify_idx + k + 4] = (gtime >> (k * 8)) & 0xff;
			}
			gtime++;
			// return pointer, i.e. current entry since we have 1024 blocks for 1024 files.
			// We allocate 32 blocks for each file.
			return fs->FILE_BASE_ADDRESS + idx * (fs->STORAGE_BLOCK_SIZE * 32); 
			
		}
	}
	// not find
	if (!find) {
		if (op == G_READ) {
			printf("ERROR: no such file.\n");
			return 0;
		} else if (op == G_WRITE) {
			fs->volume[empty_entry] = 1; // change status of FCB in super block
			// write file name in FCB
			tmp = *s;
			j = 0;
			while (tmp != '\0') {
				fs->volume[fs->SUPERBLOCK_SIZE + empty_entry * fs->FCB_SIZE + j] = tmp;
				j++;
				tmp = *(s + j);
			}
			fs->volume[fs->SUPERBLOCK_SIZE + empty_entry * fs->FCB_SIZE + j] = tmp;

			// add create time and modified time
			int create_idx = fs->SUPERBLOCK_SIZE + empty_entry * fs->FCB_SIZE + fs->MAX_FILENAME_SIZE;
			for (int k = 0; k < 4; k++) {
				fs->volume[create_idx + k] = (gtime >> (k * 8)) & 0xff;
				fs->volume[create_idx + k + 4] = (gtime >> (k * 8)) & 0xff;
				fs->volume[create_idx + k + 8] = (0 >> (k * 8)) & 0xff;
			}
			gtime++;
			
			return fs->FILE_BASE_ADDRESS + empty_entry * (fs->STORAGE_BLOCK_SIZE * 32);
			

		} else {
			printf("ERROR: invalid operation.\n");
			return 0;
		}
		

	}
}


__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp)
{
	/* Implement read operation here */
	for (int i = fp; i < fp + size; i++) {
		*(output + i) = fs->volume[i];
	}
}

__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp)
{
	/* Implement write operation here */
	for (int i = fp; i < fp + size; i++) {
		fs->volume[i] = *(input + i);
	}
	// Record size of the file in FCB
	int FCB_idx = ((fp - fs->FILE_BASE_ADDRESS) / (fs->STORAGE_BLOCK_SIZE * 32)) * fs->FCB_SIZE + fs->SUPERBLOCK_SIZE;
	for (int k = 0; k < 4; k++) {
		fs->volume[FCB_idx + fs->MAX_FILENAME_SIZE + k + 8] = (size >> (k * 8)) & 0xff;
	}
	return 0;

}
__device__ void fs_gsys(FileSystem *fs, int op)
{
	/* Implement LS_D and LS_S operation here */
	if (op == LS_D) {
		printf("===sort by modified time===\n");
		int tmax = 9999;
		for (int t = 0; t < fs->FCB_ENTRIES; t++) {
			if (fs->volume[t] != 1) {
				continue;
			}
			int tmpmax = -1;
			int argmax;
			for (int i = 0; i < fs->FCB_ENTRIES; i++) {
				if (fs->volume[i] == 1) {

					int idx = fs->SUPERBLOCK_SIZE + i * fs->FCB_SIZE;


					int created_time = 0;
					int modified_time = 0;
					int size = 0;
					for (int k = 0; k < 4; k++) {
						created_time += fs->volume[idx + fs->MAX_FILENAME_SIZE + k] << (k * 8);
						modified_time += fs->volume[idx + fs->MAX_FILENAME_SIZE + k + 4] << (k * 8);
						size += fs->volume[idx + fs->MAX_FILENAME_SIZE + k + 8] << (k * 8);
					}
					if (tmpmax < modified_time && modified_time < tmax) {
						tmpmax = modified_time;
						argmax = idx;
					}
				}
			}
			int j = 0;
			while (fs->volume[argmax + j] != '\0') {
				printf("%c", fs->volume[argmax + j]);
				j++;
			}
			printf("\n", tmpmax);
			tmax = tmpmax;
		}
	} else if (op == LS_S) {
		printf("===sort by file size===\n");
		int tmax = 9999;
		int lasttime = -1;
		for (int t = 0; t < fs->FCB_ENTRIES; t++) {
			if (fs->volume[t] != 1) {
				continue;
			}
			int tmpmax = -1;
			int argmax;
			int tmptime = -1;
			for (int i = 0; i < fs->FCB_ENTRIES; i++) {
				if (fs->volume[i] == 1) {

					int idx = fs->SUPERBLOCK_SIZE + i * fs->FCB_SIZE;


					int created_time = 0;
					int modified_time = 0;
					int size = 0;
					for (int k = 0; k < 4; k++) {
						created_time += fs->volume[idx + fs->MAX_FILENAME_SIZE + k] << (k * 8);
						modified_time += fs->volume[idx + fs->MAX_FILENAME_SIZE + k + 4] << (k * 8);
						size += fs->volume[idx + fs->MAX_FILENAME_SIZE + k + 8] << (k * 8);
					}
					if (size > tmpmax) {
						if (size < tmax) {
							tmpmax = size;
							argmax = idx;
							tmptime = created_time;
						}
						else if (size == tmax) {
							if (created_time > lasttime) {
								tmpmax = size;
								argmax = idx;
								tmptime = created_time;
							}
						}
					}
					else if (size == tmpmax) {
						if (created_time < tmptime) {
							tmpmax = size;
							argmax = idx;
							tmptime = created_time;
						}
					}

				}
			}
			int j = 0;
			while (fs->volume[argmax + j] != '\0') {
				printf("%c", fs->volume[argmax + j]);
				j++;
			}
			printf(" %d\n", tmpmax);
			tmax = tmpmax;
			lasttime = tmptime; 

		}
	} else {
		printf("ERROR: invalid operation\n");
		return;
	}


}

__device__ void fs_gsys(FileSystem *fs, int op, char *s)
{
	/* Implement rm operation here */
	if (op != RM) {
		printf("ERROR: not valid operation");
		return;
	}
	int i = fs->SUPERBLOCK_SIZE;
	bool find = false;
	char tmp;
	int j;
	int idx;
	bool flag;
	while (i < fs->SUPERBLOCK_SIZE + fs->FCB_ENTRIES * fs->FCB_SIZE) {
		idx = (i - fs->SUPERBLOCK_SIZE) / fs->FCB_SIZE;
		if (fs->volume[idx] != 1) {
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
			//printf("this block = %d\n", i);
			//printf("not find in this block\n");
			i += fs->FCB_SIZE;
			continue;
		}
		else
		{
			find = true;
			fs->volume[idx] = 0;
		}
	}
}
