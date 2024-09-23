/*
    1.IO模式: N-N
    2.读 or 写
    3.读写并行度
    4.网格维度、大小
    5.每次读写步长
    6.指定文件名 (可选)
    7.迭代次数
    8.间隔描述
*/  

#include <mpi.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <cstring>
#include <sys/stat.h>
#include <fcntl.h>
#include <algorithm>
#include <hdf5.h>

using namespace std;

void HDF5_NtoN_write_2D(int rank, int size, string filename, vector<int> grid_dims_size, int file_count, int step, bool keep_files)
{
    hsize_t NX = grid_dims_size[0], NY = grid_dims_size[1];
    hsize_t STEP_X = step;

    for(int i = 0; i < file_count; i++){
        filename = filename + to_string(rank) + "_" + to_string(i) + ".h5";
        hid_t file_id, dataspace_id, dataset_id, prop_id;
        herr_t status;
        hsize_t dims[2] = {NX, NY}; 
        hsize_t chunk_dims[2] = {STEP_X, NY}; 
        hsize_t start[2]; 
        hsize_t count[2] = {STEP_X, NY};

        file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

        dataspace_id = H5Screate_simple(2, dims, NULL);

        prop_id = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(prop_id, 2, chunk_dims);

        dataset_id = H5Dcreate2(file_id, "data2D", H5T_NATIVE_DOUBLE, dataspace_id, H5P_DEFAULT, prop_id, H5P_DEFAULT);

        double *data = new double[STEP_X * NY];
        memset(data, rank, sizeof(double) * STEP_X * NY);

        for(int x = 0; x < NX; x += STEP_X) {
            start[0] = x; start[1] = 0;
            status = H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, start, NULL, count, NULL);

            hid_t memspace_id = H5Screate_simple(2, count, NULL);
            status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, memspace_id, dataspace_id, H5P_DEFAULT, data);
            H5Sclose(memspace_id);
        }

        delete[] data;
        H5Dclose(dataset_id);
        H5Sclose(dataspace_id);
        H5Pclose(prop_id);
        H5Fclose(file_id);

        if(!keep_files)
            remove(filename.c_str());
    }
}

void HDF5_NtoN_read_2D(int rank, int size, string filename, vector<int> grid_dims_size, int file_count, int step, bool keep_files)
{
    hsize_t NX = grid_dims_size[0], NY = grid_dims_size[1];
    hsize_t STEP_X = step;

    for(int i = 0; i < file_count; i++){   
        filename = filename + to_string(rank) + "_" + to_string(i) + ".h5";
        hid_t file_id, dataspace_id, dataset_id, prop_id, memspace_id;
        herr_t status;
        hsize_t dims[2] = {NX, NY};     
        hsize_t start[2];
        hsize_t count[2] = {STEP_X, NY};

        file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        dataset_id = H5Dopen(file_id, "data2D", H5P_DEFAULT);
        dataspace_id = H5Dget_space(dataset_id);

        double *data = new double[STEP_X * NY];

        for(int x = 0; x < NX; x += STEP_X) {
            start[0] = x; start[1] = 0;
            status = H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, start, NULL, count, NULL);

            memspace_id = H5Screate_simple(2, count, NULL);

            status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, memspace_id, dataspace_id, H5P_DEFAULT, data);

            H5Sclose(memspace_id);
        }

        delete[] data;
        H5Dclose(dataset_id);
        H5Sclose(dataspace_id);
        H5Fclose(file_id);  
        if(!keep_files)
            remove(filename.c_str());
    }
}

void HDF5_NtoN_write_3D(int rank, int size, string filename, vector<int> grid_dims_size, int file_count, int step, bool keep_files)
{
    hsize_t NX = grid_dims_size[0], NY = grid_dims_size[1], NZ = grid_dims_size[2];
    hsize_t STEP_X = step;

    for(int i = 0; i < file_count; i++){
        filename = filename + to_string(rank) + "_" + to_string(i) + ".h5";
        hid_t file_id, dataspace_id, dataset_id, prop_id;
        herr_t status;
        hsize_t dims[3] = {NX, NY, NZ}; 
        hsize_t chunk_dims[3] = {STEP_X, NY, NZ}; 
        hsize_t start[3]; 
        hsize_t count[3] = {STEP_X, NY, NZ};

        file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

        dataspace_id = H5Screate_simple(3, dims, NULL);

        prop_id = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(prop_id, 3, chunk_dims);

        dataset_id = H5Dcreate2(file_id, "data3D", H5T_NATIVE_DOUBLE, dataspace_id, H5P_DEFAULT, prop_id, H5P_DEFAULT);

        double *data = new double[STEP_X * NY * NZ];
        memset(data, rank, sizeof(double) * STEP_X * NY * NZ);

        for(int x = 0; x < NX; x += STEP_X) {
            start[0] = x; start[1] = 0; start[2] = 0;
            status = H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, start, NULL, count, NULL);

            hid_t memspace_id = H5Screate_simple(3, count, NULL);
            status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, memspace_id, dataspace_id, H5P_DEFAULT, data);
            H5Sclose(memspace_id);
        }

        delete[] data;
        H5Dclose(dataset_id);
        H5Sclose(dataspace_id);
        H5Pclose(prop_id);
        H5Fclose(file_id);
        if(!keep_files)
            remove(filename.c_str());
    }
}

void HDF5_NtoN_read_3D(int rank, int size, string filename, vector<int> grid_dims_size, int file_count, int step, bool keep_files)
{

    hsize_t NX = grid_dims_size[0], NY = grid_dims_size[1], NZ = grid_dims_size[2];
    hsize_t STEP_X = step;

    for(int i = 0; i < file_count; i++){
        filename = filename + to_string(rank) + "_" + to_string(i) + ".h5";
        hid_t file_id, dataspace_id, dataset_id, prop_id, memspace_id;
        herr_t status;
        hsize_t dims[3] = {NX, NY, NZ}; 
        hsize_t start[3];
        hsize_t count[3] = {STEP_X, NY, NZ};

        file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        dataset_id = H5Dopen(file_id, "data3D", H5P_DEFAULT);
        dataspace_id = H5Dget_space(dataset_id);

        double *data = new double[STEP_X * NY * NZ];

        for(int x = 0; x < NX; x += STEP_X) {
            start[0] = x; start[1] = 0; start[2] = 0;
            status = H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, start, NULL, count, NULL);

            memspace_id = H5Screate_simple(3, count, NULL);

            status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, memspace_id, dataspace_id, H5P_DEFAULT, data);

            H5Sclose(memspace_id);
        }

        delete[] data;
        H5Dclose(dataset_id);
        H5Sclose(dataspace_id);
        H5Fclose(file_id);
        if(!keep_files)
            remove(filename.c_str());
    }

}

void HDF5_NtoN_write_4D(int rank, int size, string filename, vector<int> grid_dims_size, int file_count, int step, bool kee_files)
{
    hsize_t NX = grid_dims_size[0], NY = grid_dims_size[1], NZ = grid_dims_size[2], NT = grid_dims_size[3];
    hsize_t STEP_X = step;

    for(int i = 0; i < file_count; i++){
        filename = filename + to_string(rank) + "_" + to_string(i) + ".h5";
        hid_t file_id, dataspace_id, dataset_id, prop_id; 
 
        herr_t status;  
        hsize_t dims[4] = {NX, NY, NZ, NT};
        
        hsize_t chunk_dims[4] = {STEP_X, NY, NZ, NT};
        hsize_t start[4];
        hsize_t count[4] = {STEP_X, NY, NZ, NT};

        file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

        dataspace_id = H5Screate_simple(4, dims, NULL);

        prop_id = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(prop_id, 4, chunk_dims);

        dataset_id = H5Dcreate2(file_id, "data4D", H5T_NATIVE_DOUBLE, dataspace_id, H5P_DEFAULT, prop_id, H5P_DEFAULT);

        double *data = new double[STEP_X * NY * NZ * NT];
        memset(data, rank, sizeof(double) * STEP_X * NY * NZ * NT);

        for(int x = 0; x < NX; x += STEP_X) {
            start[0] = x; start[1] = 0; start[2] = 0; start[3] = 0;
            status = H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, start, NULL, count, NULL);

            hid_t memspace_id = H5Screate_simple(4, count, NULL);
            status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, memspace_id, dataspace_id, H5P_DEFAULT, data);
            H5Sclose(memspace_id);
        }

        delete[] data;
        H5Dclose(dataset_id);
        H5Sclose(dataspace_id);
        H5Pclose(prop_id);
        H5Fclose(file_id);
        remove(filename.c_str());
    }
}

void HDF5_NtoN_read_4D(int rank, int size, string filename, vector<int> grid_dims_size, int file_count, int step, bool keep_files)
{
    hsize_t NX = grid_dims_size[0], NY = grid_dims_size[1], NZ = grid_dims_size[2], NT = grid_dims_size[3];
    hsize_t STEP_X = step;

    for(int i = 0; i < file_count; i++){   
    
        filename = filename + to_string(rank) + "_" + to_string(i) + ".h5";
        hid_t file_id, dataspace_id, dataset_id, prop_id, memspace_id;
        herr_t status;
        hsize_t dims[4] = {NX, NY, NZ, NT};
        hsize_t start[4];
        hsize_t count[4] = {STEP_X, NY, NZ, NT};

        file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        dataset_id = H5Dopen(file_id, "data4D", H5P_DEFAULT);
        dataspace_id = H5Dget_space(dataset_id);

        double *data = new double[STEP_X * NY * NZ * NT];

        for(int x = 0; x < NX; x += STEP_X) {
            start[0] = x; start[1] = 0; start[2] = 0; start[3] = 0;
            status = H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, start, NULL, count, NULL);

            memspace_id = H5Screate_simple(4, count, NULL);

            status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, memspace_id, dataspace_id, H5P_DEFAULT, data);

            H5Sclose(memspace_id);
        }

        delete[] data;
        H5Dclose(dataset_id);
        H5Sclose(dataspace_id);
        H5Fclose(file_id);
        if(!keep_files)
            remove(filename.c_str());
    }
}

void HDF5_Nto1_write_2D(int rank, int size, string filename, vector<int> grid_dims_size, int step, bool keep_files)
{
    string file_name = filename + ".h5";

    hsize_t NX = grid_dims_size[0], NY = grid_dims_size[1];
    hsize_t STEP_X = step;

    hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);

    hid_t file_id = H5Fcreate(file_name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
    H5Pclose(plist_id);

    hsize_t global_dims[2] = {NX * size, NY};
    hid_t space_id = H5Screate_simple(2, global_dims, NULL);

    hid_t dset_id = H5Dcreate(file_id, "data2D", H5T_NATIVE_DOUBLE, space_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    hsize_t count[2] = {NX, NY};
    hsize_t start[2] = {rank * NX, 0};
    hid_t memspace_id = H5Screate_simple(2, count, NULL);

    H5Sselect_hyperslab(space_id, H5S_SELECT_SET, start, NULL, count, NULL);

    hid_t xfer_plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(xfer_plist_id, H5FD_MPIO_COLLECTIVE);

    vector<double> data(grid_dims_size[0] * grid_dims_size[1], rank + 1.0);

    for(int x_offset = 0; x_offset < grid_dims_size[0]; x_offset += step) {
        hsize_t current_step = min(step, grid_dims_size[0] - x_offset);
        vector<double> current_data(current_step * grid_dims_size[1], rank + 1.0);
        hsize_t chunk_start[2] = {start[0] + x_offset, start[1]} ;
        hsize_t chunk_count[2] = {current_step, NY};
        
        memspace_id = H5Screate_simple(2, chunk_count, NULL);
        H5Sselect_hyperslab(space_id, H5S_SELECT_SET, chunk_start, NULL, chunk_count, NULL);
        H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, memspace_id, space_id, xfer_plist_id, current_data.data());
    }

    // Close all handles and release resources
    H5Dclose(dset_id);
    H5Sclose(space_id);
    H5Sclose(memspace_id);
    H5Pclose(xfer_plist_id);
    H5Fclose(file_id);

    // Optionally remove the file
    if(!keep_files)
        remove(file_name.c_str());
}

void HDF5_Nto1_read_2D(int rank, int size, string filename, vector<int> grid_dims_size, int step, bool keep_files)
{
    string file_name = filename + ".h5";
    hsize_t NX = grid_dims_size[0], NY = grid_dims_size[1];
    hsize_t STEP_X = step;

    hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);

    hid_t file_id = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, plist_id);
    H5Pclose(plist_id);

    hid_t dset_id = H5Dopen(file_id, "data2D", H5P_DEFAULT);
    hid_t space_id = H5Dget_space(dset_id);

    hsize_t count[2] = {NX, NY};
    hsize_t start[2] = {rank * NX, 0};
    H5Sselect_hyperslab(space_id, H5S_SELECT_SET, start, NULL, count, NULL);

    hid_t memspace_id = H5Screate_simple(2, count, NULL);

    hid_t xfer_plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(xfer_plist_id, H5FD_MPIO_COLLECTIVE);

    vector<double> data(grid_dims_size[0] * grid_dims_size[1], 0.0);

    for(int x_offset = 0; x_offset < grid_dims_size[0]; x_offset += step) {
        hsize_t current_step = min(step, grid_dims_size[0] - x_offset);
        vector<double> current_data(current_step * grid_dims_size[1], 0.0);
        hsize_t chunk_start[2] = {start[0] + x_offset, start[1]};
        hsize_t chunk_count[2] = {current_step, NY};
        
        memspace_id = H5Screate_simple(2, chunk_count, NULL);
        H5Sselect_hyperslab(space_id, H5S_SELECT_SET, chunk_start, NULL, chunk_count, NULL);
        H5Dread(dset_id, H5T_NATIVE_DOUBLE, memspace_id, space_id, xfer_plist_id, current_data.data());
    }

    H5Dclose(dset_id);
    H5Sclose(space_id); 
    H5Sclose(memspace_id);
    H5Pclose(xfer_plist_id);
    H5Fclose(file_id);

    if(!keep_files){
        remove(file_name.c_str());  
    }
}

void HDF5_Nto1_write_3D(int rank, int size, string filename, vector<int> grid_dims_size, int step, bool keep_files)
{
    string file_name = filename + ".h5";
    hsize_t NX = grid_dims_size[0], NY = grid_dims_size[1], NZ = grid_dims_size[2];
    hsize_t STEP_X = step;

    hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);

    hid_t file_id = H5Fcreate(file_name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
    H5Pclose(plist_id);

    hsize_t global_dims[3] = {NX * size, NY, NZ};
    hid_t space_id = H5Screate_simple(3, global_dims, NULL);

    hid_t dset_id = H5Dcreate(file_id, "data3D", H5T_NATIVE_DOUBLE, space_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    hsize_t count[3] = {NX, NY, NZ};
    hsize_t start[3] = {rank * NX, 0, 0};
    hid_t memspace_id = H5Screate_simple(3, count, NULL);

    H5Sselect_hyperslab(space_id, H5S_SELECT_SET, start, NULL, count, NULL);

    hid_t xfer_plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(xfer_plist_id, H5FD_MPIO_COLLECTIVE);

    vector<double> data(grid_dims_size[0] * grid_dims_size[1] * grid_dims_size[2], rank + 1.0);

    for(int x_offset = 0; x_offset < grid_dims_size[0]; x_offset += step) {
        hsize_t current_step = min(step, grid_dims_size[0] - x_offset);
        vector<double> current_data(current_step * grid_dims_size[1] * grid_dims_size[2], rank + 1.0);
        hsize_t chunk_start[3] = {start[0] + x_offset, start[1], start[2]};
        hsize_t chunk_count[3] = {current_step, NY, NZ};
        
        memspace_id = H5Screate_simple(3, chunk_count, NULL);
        H5Sselect_hyperslab(space_id, H5S_SELECT_SET, chunk_start, NULL, chunk_count, NULL);
        H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, memspace_id, space_id, xfer_plist_id, current_data.data());
    }

    H5Dclose(dset_id);
    H5Sclose(space_id);
    H5Sclose(memspace_id);
    H5Pclose(xfer_plist_id);
    H5Fclose(file_id);

    if(!keep_files){
        remove(file_name.c_str());
    }
}

void HDF5_Nto1_read_3D(int rank, int size, string filename, vector<int> grid_dims_size, int step, bool keep_files)
{
    string file_name = filename + ".h5";
    hsize_t NX = grid_dims_size[0], NY = grid_dims_size[1], NZ = grid_dims_size[2];
    hsize_t STEP_X = step;

    hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);

    hid_t file_id = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, plist_id);
    H5Pclose(plist_id);

    hid_t dset_id = H5Dopen(file_id, "data3D", H5P_DEFAULT);
    hid_t space_id = H5Dget_space(dset_id);

    hsize_t count[3] = {NX, NY, NZ};
    hsize_t start[3] = {rank * NX, 0, 0};
    H5Sselect_hyperslab(space_id, H5S_SELECT_SET, start, NULL, count, NULL);

    hid_t memspace_id = H5Screate_simple(3, count, NULL);

    hid_t xfer_plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(xfer_plist_id, H5FD_MPIO_COLLECTIVE);

    vector<double> data(grid_dims_size[0] * grid_dims_size[1] * grid_dims_size[2], 0.0);

    for(int x_offset = 0; x_offset < grid_dims_size[0]; x_offset += step) {
        hsize_t current_step = min(step, grid_dims_size[0] - x_offset);
        vector<double> current_data(current_step * grid_dims_size[1] * grid_dims_size[2], 0.0);
        hsize_t chunk_start[3] = {start[0] + x_offset, start[1], start[2]};
        hsize_t chunk_count[3] = {current_step, NY, NZ};
        
        memspace_id = H5Screate_simple(3, chunk_count, NULL);
        H5Sselect_hyperslab(space_id, H5S_SELECT_SET, chunk_start, NULL, chunk_count, NULL);
        H5Dread(dset_id, H5T_NATIVE_DOUBLE, memspace_id, space_id, xfer_plist_id, current_data.data());
    }

    H5Dclose(dset_id);
    H5Sclose(space_id);
    H5Sclose(memspace_id);  
    H5Pclose(xfer_plist_id);
    H5Fclose(file_id);

    if(!keep_files){
        remove(file_name.c_str());
    }
}

void HDF5_Nto1_write_4D(int rank, int size, string filename, vector<int> grid_dims_size, int step, bool keep_files)
{
    string file_name = filename + ".h5";
    hsize_t NX = grid_dims_size[0], NY = grid_dims_size[1], NZ = grid_dims_size[2], NT = grid_dims_size[3];
    hsize_t STEP_X = step;

    hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);

    H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);

    hid_t file_id = H5Fcreate(file_name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
    H5Pclose(plist_id);

    hsize_t global_dims[4] = {NX * size, NY, NZ, NT};
    hid_t space_id = H5Screate_simple(4, global_dims, NULL);

    hid_t dset_id = H5Dcreate(file_id, "data4D", H5T_NATIVE_DOUBLE, space_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    hsize_t count[4] = {NX, NY, NZ, NT};
    hsize_t start[4] = {rank * NX, 0, 0, 0};
    hid_t memspace_id = H5Screate_simple(4, count, NULL);

    H5Sselect_hyperslab(space_id, H5S_SELECT_SET, start, NULL, count, NULL);

    hid_t xfer_plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(xfer_plist_id, H5FD_MPIO_COLLECTIVE);

    vector<double> data(grid_dims_size[0] * grid_dims_size[1] * grid_dims_size[2] * grid_dims_size[3], rank + 1.0);

    for(int x_offset = 0; x_offset < grid_dims_size[0]; x_offset += step) {
        hsize_t current_step = min(step, grid_dims_size[0] - x_offset);
        vector<double> current_data(current_step * grid_dims_size[1] * grid_dims_size[2] * grid_dims_size[3], rank + 1.0); 
        hsize_t chunk_start[4] = {start[0] + x_offset, start[1], start[2], start[3]};
        hsize_t chunk_count[4] = {current_step, NY, NZ, NT};
        
        memspace_id = H5Screate_simple(4, chunk_count, NULL);
        H5Sselect_hyperslab(space_id, H5S_SELECT_SET, chunk_start, NULL, chunk_count, NULL);
        H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, memspace_id, space_id, xfer_plist_id, current_data.data());
    }

    H5Dclose(dset_id);
    H5Sclose(space_id);
    H5Sclose(memspace_id);
    H5Pclose(xfer_plist_id);
    H5Fclose(file_id);

    if(!keep_files){
        remove(file_name.c_str());
    }

}

void HDF5_Nto1_read_4D(int rank, int size, string filename, vector<int> grid_dims_size, int step, bool keep_files)
{

    string file_name = filename + ".h5";
    hsize_t NX = grid_dims_size[0], NY = grid_dims_size[1], NZ = grid_dims_size[2], NT = grid_dims_size[3];
    hsize_t STEP_X = step;

    hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);

    hid_t file_id = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, plist_id);
    H5Pclose(plist_id);

    hid_t dset_id = H5Dopen(file_id, "data4D", H5P_DEFAULT);
    hid_t space_id = H5Dget_space(dset_id);

    hsize_t count[4] = {NX, NY, NZ, NT};
    hsize_t start[4] = {rank * NX, 0, 0, 0};
    H5Sselect_hyperslab(space_id, H5S_SELECT_SET, start, NULL, count, NULL);

    hid_t memspace_id = H5Screate_simple(4, count, NULL);

    hid_t xfer_plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(xfer_plist_id, H5FD_MPIO_COLLECTIVE);

    vector<double> data(grid_dims_size[0] * grid_dims_size[1] * grid_dims_size[2] * grid_dims_size[3], 0.0);

    for(int x_offset = 0; x_offset < grid_dims_size[0]; x_offset += step) {
        hsize_t current_step = min(step, grid_dims_size[0] - x_offset);
        vector<double> current_data(current_step * grid_dims_size[1] * grid_dims_size[2] * grid_dims_size[3], 0.0);
        hsize_t chunk_start[4] = {start[0] + x_offset, start[1], start[2], start[3]};
        hsize_t chunk_count[4] = {current_step, NY, NZ, NT};
        
        memspace_id = H5Screate_simple(4, chunk_count, NULL);
        H5Sselect_hyperslab(space_id, H5S_SELECT_SET, chunk_start, NULL, chunk_count, NULL);
        H5Dread(dset_id, H5T_NATIVE_DOUBLE, memspace_id, space_id, xfer_plist_id, current_data.data());

    }

    H5Dclose(dset_id);
    H5Sclose(space_id);
    H5Sclose(memspace_id);
    H5Pclose(xfer_plist_id);
    H5Fclose(file_id);

    if(!keep_files){
        remove(file_name.c_str());
    }
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    //Default pattern：N-N
    string pattern = "N-N";
    //Default op_type
    char op_type = 'w'; 
    // Default file count: 1
    int file_count = 1; 
    //Default dim
    int dimids = 3;
    //Default grid_dims_sizes
    vector<int> grid_dims_size;
    //Default step
    int step = 1;
    // Default filename
    string filename = "hdf5";
    // Default keep
    bool keep_files = false;
    // Default iterations
    int iterations = 1;
    // Default sleep
    int sleep_seconds = 0;

    // Parse command line arguments
    for (int i = 1; i < argc; i++){

        if(strcmp(argv[i], "-p") == 0){
            if(strcmp(argv[i + 1], "N-1") == 0){
                pattern = "N-1";
            }
        }
        else if(strcmp(argv[i], "-n") == 0){
            file_count = atoi(argv[i + 1]);
        }
        else if(strcmp(argv[i], "-f") == 0){
            filename = argv[i + 1];
        }
        else if(strcmp(argv[i], "-r") == 0){
            op_type = 'r';
        }
        else if(strcmp(argv[i], "-d") == 0){
            dimids = atoi(argv[i + 1]);
            for(int j = 1; j <= dimids; j++)
                grid_dims_size.push_back(atoi(argv[i + 1 + j]));
        }
        else if(strcmp(argv[i], "-s") == 0){
            step = atoi(argv[i + 1]);
        }
        else if(strcmp(argv[i], "-k") == 0){
            keep_files = true;
        }
        else if(strcmp(argv[i], "-i") == 0){
            iterations = atoi(argv[i + 1]);
        } 
        else if(strcmp(argv[i], "-g") == 0){
            sleep_seconds = atoi(argv[i + 1]);
        }
    }


    for(int i = 0; i < iterations; i++){
	std::cout << "iterations: " << i << std::endl;
        if(op_type == 'w'){
            if(pattern == "N-N"){
                if(dimids == 2){
                    HDF5_NtoN_write_2D(rank, size, filename, grid_dims_size, file_count, step, keep_files);
                }
                else if(dimids == 3){
                    HDF5_NtoN_write_3D(rank, size, filename, grid_dims_size, file_count, step, keep_files);
                }
                else if(dimids == 4){
                    HDF5_NtoN_write_4D(rank, size, filename, grid_dims_size, file_count, step, keep_files);
                }
            }
            else if(pattern == "N-1"){
                if(dimids == 2){
                    HDF5_Nto1_write_2D(rank, size, filename, grid_dims_size, step, keep_files); 
                }
                else if(dimids == 3){
                    HDF5_Nto1_write_3D(rank, size, filename, grid_dims_size, step, keep_files);
                }
                else if(dimids == 4){
                    HDF5_Nto1_write_4D(rank, size, filename, grid_dims_size, step, keep_files);
                }
            }
            
        }

        else if(op_type == 'r'){
            if(pattern == "N-N"){
                if(dimids == 2){
                    HDF5_NtoN_read_2D(rank, size, filename, grid_dims_size, file_count, step, keep_files);
                }
                else if(dimids == 3){
                    HDF5_NtoN_read_3D(rank, size, filename, grid_dims_size, file_count, step, keep_files);
                }
                else if(dimids == 4){
                    HDF5_NtoN_read_4D(rank, size, filename, grid_dims_size, file_count, step, keep_files);
                }
            }
            else if(pattern == "N-1"){
                if(dimids == 2){
                    HDF5_Nto1_read_2D(rank, size, filename, grid_dims_size, step, keep_files);
                }
                else if(dimids == 3){
                    HDF5_Nto1_read_3D(rank, size, filename, grid_dims_size, step, keep_files);
                }
                else if(dimids == 4){
                    HDF5_Nto1_read_4D(rank, size, filename, grid_dims_size, step, keep_files);
                }
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        sleep(sleep_seconds);
    }

    MPI_Finalize();

    return 0;

}
