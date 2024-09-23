/*
    1.IO模式: N-1
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
#include <pnetcdf.h>

using namespace std;

#define HANDLE_ERROR(err) { \
    if ((err) != NC_NOERR) { \
        std::cerr << "PnetCDF error: " << ncmpi_strerror(err) << std::endl; \
        MPI_Abort(MPI_COMM_WORLD, -1); \
    } \
}

void NetCDF_Nto1_write_2D(int rank, int size, string filename, vector<int> grid_dims_size, int chunk_size, bool keep_files)
{
    int ncid, varid, dimids[2];
    int local_nx = grid_dims_size[0], NY = grid_dims_size[1];

    MPI_Offset start[2], count[2];

    int global_nx = local_nx * size;
    start[1] = 0;
    count[1] = NY;

    filename += ".nc";
    HANDLE_ERROR(ncmpi_create(MPI_COMM_WORLD, filename.c_str(), NC_CLOBBER, MPI_INFO_NULL, &ncid));

    ncmpi_def_dim(ncid, "x", global_nx, &dimids[0]);
    ncmpi_def_dim(ncid, "y", NY, &dimids[1]);
    ncmpi_def_var(ncid, "data", NC_DOUBLE, 2, dimids, &varid);
    ncmpi_enddef(ncid);

    start[0] = rank * local_nx;
    count[0] = local_nx;

    for(int x_offset = 0; x_offset < local_nx; x_offset += chunk_size){
        int current_chunk_size = min(chunk_size, local_nx - x_offset);
        vector<double> chunk_data(current_chunk_size * NY, rank + 1.0d);

        MPI_Offset chunk_start[2] = {start[0] + x_offset, start[1]};
        MPI_Offset chunk_count[2] = {current_chunk_size, count[1]};

        ncmpi_put_vara_double_all(ncid, varid, chunk_start, chunk_count, chunk_data.data());
    }

    ncmpi_close(ncid);
    if(!keep_files)
        remove(filename.c_str());
}

void NetCDF_Nto1_read_2D(int rank, int size, string filename, vector<int> grid_dims_size, int chunk_size, bool keep_files) 
{
    int ncid, varid, dimids[2];
    int local_nx = grid_dims_size[0], NY = grid_dims_size[1];

    MPI_Offset start[2], count[2];

    int global_nx = local_nx * size;
    start[1] = 0;
    count[1] = NY;

    filename += ".nc";
    ncmpi_open(MPI_COMM_WORLD, filename.c_str(), NC_NOWRITE, MPI_INFO_NULL, &ncid);

    ncmpi_inq_varid(ncid, "data", &varid);

    start[0] = rank * local_nx;
    count[0] = local_nx;

    std::vector<double> data(local_nx * NY);

    for (int x_offset = 0; x_offset < local_nx; x_offset += chunk_size) {
        int current_chunk_size = std::min(chunk_size, local_nx - x_offset);
        vector<double> chunk_data(current_chunk_size * NY);

        MPI_Offset chunk_start[2] = {start[0] + x_offset, start[1]};
        MPI_Offset chunk_count[2] = {current_chunk_size, count[1]};

        ncmpi_get_vara_double_all(ncid, varid, chunk_start, chunk_count, chunk_data.data());
    }

    ncmpi_close(ncid);
    if(!keep_files)
        remove(filename.c_str());
}

void NetCDF_Nto1_write_3D(int rank, int size, string filename, vector<int> grid_dims_size, int chunk_size, bool keep_files)
{
    int ncid, varid, dimids[3];
    int local_nx = grid_dims_size[0], NY = grid_dims_size[1], NZ = grid_dims_size[2];

    MPI_Offset start[3], count[3];

    int global_nx = local_nx * size;
    start[1] = start[2] = 0;
    count[1] = NY;
    count[2] = NZ;

    filename += ".nc";
    ncmpi_create(MPI_COMM_WORLD, filename.c_str(), NC_CLOBBER, MPI_INFO_NULL, &ncid);

    ncmpi_def_dim(ncid, "x", global_nx, &dimids[0]);
    ncmpi_def_dim(ncid, "y", NY, &dimids[1]);
    ncmpi_def_dim(ncid, "z", NZ, &dimids[2]);
    ncmpi_def_var(ncid, "data", NC_DOUBLE, 3, dimids, &varid);
    ncmpi_enddef(ncid);

    start[0] = rank * local_nx;
    count[0] = local_nx;

    for(int x_offset = 0; x_offset < local_nx; x_offset += chunk_size){
        int current_chunk_size = min(chunk_size, local_nx - x_offset);
        vector<double> chunk_data(current_chunk_size * NY * NZ, rank + 1.0d);

        MPI_Offset chunk_start[3] = {start[0] + x_offset, start[1], start[2]};
        MPI_Offset chunk_count[3] = {current_chunk_size, count[1], count[2]};

        ncmpi_put_vara_double_all(ncid, varid, chunk_start, chunk_count, chunk_data.data());
    }
    ncmpi_close(ncid);  
    if(!keep_files)
        remove(filename.c_str());
}

void NetCDF_Nto1_read_3D(int rank, int size, string filename, vector<int> grid_dims_size, int chunk_size, bool keep_files) 
{
    int ncid, varid, dimids[3];
    int local_nx = grid_dims_size[0], NY = grid_dims_size[1], NZ = grid_dims_size[2];

    MPI_Offset start[3], count[3];

    int global_nx = local_nx * size;
    start[1] = start[2] = 0;
    count[1] = NY;
    count[2] = NZ;

    filename += ".nc";
    ncmpi_open(MPI_COMM_WORLD, filename.c_str(), NC_NOWRITE, MPI_INFO_NULL, &ncid);

    ncmpi_inq_varid(ncid, "data", &varid);

    start[0] = rank * local_nx;
    count[0] = local_nx;
    for (int x_offset = 0; x_offset < local_nx; x_offset += chunk_size) {
        int current_chunk_size = std::min(chunk_size, local_nx - x_offset);
        vector<double> chunk_data(current_chunk_size * NY * NZ);

        MPI_Offset chunk_start[3] = {start[0] + x_offset, start[1], start[2]};
        MPI_Offset chunk_count[3] = {current_chunk_size, count[1], count[2]};

        ncmpi_get_vara_double_all(ncid, varid, chunk_start, chunk_count, chunk_data.data());
    }

    ncmpi_close(ncid);
    if(!keep_files)
        remove(filename.c_str());
}

void NetCDF_Nto1_write_4D(int rank, int size, string filename, vector<int> grid_dims_size, int chunk_size, bool keep_files)
{
    int ncid, varid, dimids[4];

    int local_nx = grid_dims_size[0], NY = grid_dims_size[1], NZ = grid_dims_size[2], NT = grid_dims_size[3];

    MPI_Offset start[4], count[4];

    int global_nx = local_nx * size;
    start[1] = start[2] = start[3] = 0;
    count[1] = NY;
    count[2] = NZ;
    count[3] = NT;      

    filename += ".nc";
    ncmpi_create(MPI_COMM_WORLD, filename.c_str(), NC_CLOBBER, MPI_INFO_NULL, &ncid);
    
    ncmpi_def_dim(ncid, "x", global_nx, &dimids[0]);
    ncmpi_def_dim(ncid, "y", NY, &dimids[1]);  
    ncmpi_def_dim(ncid, "z", NZ, &dimids[2]);
    ncmpi_def_dim(ncid, "t", NT, &dimids[3]);

    ncmpi_def_var(ncid, "data", NC_DOUBLE, 4, dimids, &varid);

    ncmpi_enddef(ncid);

    start[0] = rank * local_nx;     
    count[0] = local_nx;   

    for(int x_offset = 0; x_offset < local_nx; x_offset += chunk_size){ 

        int current_chunk_size = min(chunk_size, local_nx - x_offset);    

        vector<double> chunk_data(current_chunk_size * NY * NZ * NT, rank + 1.0d);

        MPI_Offset chunk_start[4] = {start[0] + x_offset, start[1], start[2], start[3]};
        MPI_Offset chunk_count[4] = {current_chunk_size, count[1], count[2], count[3]};

        ncmpi_put_vara_double_all(ncid, varid, chunk_start, chunk_count, chunk_data.data());
    }
    ncmpi_close(ncid);  
    if(!keep_files)
        remove(filename.c_str());
}

void NetCDF_Nto1_read_4D(int rank, int size, string filename, vector<int> grid_dims_size, int chunk_size, bool keep_files)
{
    int ncid, varid, dimids[4];

    int local_nx = grid_dims_size[0], NY = grid_dims_size[1], NZ = grid_dims_size[2], NT = grid_dims_size[3];

    MPI_Offset start[4], count[4];        

    int global_nx = local_nx * size;
    start[1] = start[2] = start[3] = 0;
    count[1] = NY;
    count[2] = NZ; 
    count[3] = NT;

    filename += ".nc";
    ncmpi_open(MPI_COMM_WORLD, filename.c_str(), NC_NOWRITE, MPI_INFO_NULL, &ncid);

    ncmpi_inq_varid(ncid, "data", &varid);

    start[0] = rank * local_nx;
    count[0] = local_nx;
    for (int x_offset = 0; x_offset < local_nx; x_offset += chunk_size) {
        int current_chunk_size = std::min(chunk_size, local_nx - x_offset);    
        vector<double> chunk_data(current_chunk_size * NY * NZ * NT);

        MPI_Offset chunk_start[4] = {start[0] + x_offset, start[1], start[2], start[3]};
        MPI_Offset chunk_count[4] = {current_chunk_size, count[1], count[2], count[3]};

        ncmpi_get_vara_double_all(ncid, varid, chunk_start, chunk_count, chunk_data.data());
    }
    ncmpi_close(ncid);
    if(!keep_files)
        remove(filename.c_str());
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    struct stat st;
    stat("/datafiles/a", &st);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    //Default pattern：N-N
    string pattern = "N-1";
    //Default op_type
    char op_type = 'w'; 
    // Default file count: 1
    int file_count = 1; 
    //Default file format
    bool is_netcdf4 = false;
    //Default dim
    int dimids = 3;
    //Default grid_dims_sizes
    vector<int> grid_dims_size;
    //Default step
    int step = 1;
    // Default filename
    string filename = "pnetcdf"; 
    // Default keep
    bool keep_files = false;
    // Default iterations
    int iterations = 1;
    // Default sleep
    int sleep_seconds = 0;

    for (int i = 1; i < argc; i++){

        if(strcmp(argv[i], "-p") == 0){
            if(strcmp(argv[i + 1], "N-1") == 0){
                pattern = "N-1";
            }
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
	std::cout << "interations: " << i << std:: endl;
        if(op_type == 'w'){
            if(pattern == "N-1"){
                if(dimids == 2){
                    cout << "2D write" << endl;
                    NetCDF_Nto1_write_2D(rank, size, filename, grid_dims_size, step, keep_files);
                }
                else if (dimids == 3){
                    NetCDF_Nto1_write_3D(rank, size, filename, grid_dims_size, step, keep_files);
                }
                else if(dimids == 4){
                    NetCDF_Nto1_write_4D(rank, size, filename, grid_dims_size, step, keep_files);
                }
            }
        }
        else if (op_type == 'r')
        {
            if(pattern == "N-1"){
                if(dimids == 2){
                    NetCDF_Nto1_read_2D(rank, size, filename, grid_dims_size, step, keep_files);
                }
                else if (dimids == 3){
                    NetCDF_Nto1_read_3D(rank, size, filename, grid_dims_size, step, keep_files);
                }
                else if(dimids == 4){
                    NetCDF_Nto1_read_4D(rank, size, filename, grid_dims_size, step, keep_files);
                }
            }
        }
    
    MPI_Barrier(MPI_COMM_WORLD);
    sleep(sleep_seconds);   
}

    MPI_Finalize(); 
    return 0;

}
