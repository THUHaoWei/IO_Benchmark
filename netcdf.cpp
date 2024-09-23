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
#include <netcdf.h>

using namespace std;

void handle_error(int status)
{
    if (status != NC_NOERR)
    {
        cerr << "NetCDF error: " << nc_strerror(status) << endl;
        exit(-1);
    }
}

void NetCDF_NtoN_write_2D(int rank, int size, string filename, bool is_netcdf4, vector<int> grid_dims_size, int file_count, int chunk_size)
{
    int ncid, x_dimid, y_dimid, varid;
    int NX = grid_dims_size[0], NY = grid_dims_size[1];

    for (int i = 0; i < file_count; i++)
    {
        string file_name = filename + to_string(rank) + "_" + to_string(i) + ".nc";

        int flags = NC_CLOBBER;
        if (is_netcdf4)
        {
            flags |= NC_NETCDF4;
        }

        handle_error(nc_create(file_name.c_str(), flags, &ncid));

        handle_error(nc_def_dim(ncid, "x", NX, &x_dimid));
        handle_error(nc_def_dim(ncid, "y", NY, &y_dimid));

        const char *var_name = "data";
        int dimids[2] = {x_dimid, y_dimid};
        handle_error(nc_def_var(ncid, var_name, NC_DOUBLE, 2, dimids, &varid));

        handle_error(nc_enddef(ncid));

        for (size_t j = 0; j < NX; j += chunk_size)
        {
            size_t start[2] = {j, 0};
            size_t count[2] = {min(static_cast<size_t>(chunk_size), static_cast<size_t>(NX) - j), static_cast<size_t>(NY)};

            vector<double> data(NY * count[0], rank);

            handle_error(nc_put_vara_double(ncid, varid, start, count, data.data()));
        }

        handle_error(nc_close(ncid));
    }
}

void NetCDF_NtoN_read_2D(int rank, int size, const string &filename, bool is_netcdf4, const vector<int> &grid_dims_size, int file_count, int chunk_size)
{
    int ncid, varid;
    int NX = grid_dims_size[0], NY = grid_dims_size[1];

    for (int i = 0; i < file_count; i++)
    {
        string file_name = filename + to_string(rank) + "_" + to_string(i) + ".nc";

        int flags = NC_NOWRITE;
        if (is_netcdf4)
        {
            flags |= NC_NETCDF4;
        }

        handle_error(nc_open(file_name.c_str(), flags, &ncid));

        handle_error(nc_inq_varid(ncid, "data", &varid)); // Get variable ID

        vector<double> data(NX * NY);

        for (size_t j = 0; j < NX; j += chunk_size)
        {
            size_t start[2] = {j, 0};
            size_t count[2] = {min(static_cast<size_t>(chunk_size), static_cast<size_t>(NX) - j), static_cast<size_t>(NY)};

            handle_error(nc_get_vara_double(ncid, varid, start, count, data.data()));
        }

        handle_error(nc_close(ncid));
    }
}

void NetCDF_NtoN_write_3D(int rank, int size, string filename, bool is_netcdf4, vector<int> grid_dims_size, int file_count, int chunk_size)
{
    int ncid, x_dimid, y_dimid, z_dimid, varid;

    int NX = grid_dims_size[0], NY = grid_dims_size[1], NZ = grid_dims_size[2];

    for (int i = 0; i < file_count; i++)
    {
        string file_name = filename + to_string(rank) + "_" + to_string(i) + ".nc";

        int flags = NC_CLOBBER;
        if (is_netcdf4)
        {
            flags |= NC_NETCDF4;
        }

        handle_error(nc_create(file_name.c_str(), flags, &ncid));

        handle_error(nc_def_dim(ncid, "x", NX, &x_dimid));
        handle_error(nc_def_dim(ncid, "y", NY, &y_dimid));
        handle_error(nc_def_dim(ncid, "z", NZ, &z_dimid));

        const char *var_name = "data";
        int dimids[3] = {x_dimid, y_dimid, z_dimid};
        handle_error(nc_def_var(ncid, var_name, NC_DOUBLE, 3, dimids, &varid));

        handle_error(nc_enddef(ncid));

        for (size_t j = 0; j < NX; j += chunk_size)
        {
            size_t start[3] = {j, 0, 0};
            size_t count[3] = {min(static_cast<size_t>(chunk_size), static_cast<size_t>(NX) - j), static_cast<size_t>(NY), static_cast<size_t>(NZ)};

            vector<double> data(count[0] * NY * NZ, rank);

            handle_error(nc_put_vara_double(ncid, varid, start, count, data.data()));
        }

        handle_error(nc_close(ncid));
    }
}

void NetCDF_NtoN_read_3D(int rank, int size, const string &filename, bool is_netcdf4, const vector<int> &grid_dims_size, int file_count, int chunk_size)
{
    int ncid, varid;
    int NX = grid_dims_size[0], NY = grid_dims_size[1], NZ = grid_dims_size[2];

    for (int i = 0; i < file_count; i++)
    {
        string file_name = filename + to_string(rank) + "_" + to_string(i) + ".nc";

        int flags = NC_NOWRITE;
        if (is_netcdf4)
        {
            flags |= NC_NETCDF4;
        }

        handle_error(nc_open(file_name.c_str(), flags, &ncid));

        handle_error(nc_inq_varid(ncid, "data", &varid)); // Get variable ID

        vector<double> data(NX * NY * NZ);

        for (size_t j = 0; j < NX; j += chunk_size)
        {
            size_t start[3] = {j, 0, 0};
            size_t count[3] = {min(static_cast<size_t>(chunk_size), static_cast<size_t>(NX) - j), static_cast<size_t>(NY), static_cast<size_t>(NZ)};

            handle_error(nc_get_vara_double(ncid, varid, start, count, data.data()));
        }

        handle_error(nc_close(ncid));
    }
}

void NetCDF_NtoN_write_4D(int rank, int size, string filename, bool is_netcdf4, vector<int> grid_dims_size, int file_count, int chunk_size)
{
    int ncid, x_dimid, y_dimid, z_dimid, t_dimid, varid;

    int NX = grid_dims_size[0], NY = grid_dims_size[1], NZ = grid_dims_size[2], NT = grid_dims_size[3];

    for (int i = 0; i < file_count; i++)
    {
        string file_name = filename + to_string(rank) + "_" + to_string(i) + ".nc";

        int flags = NC_CLOBBER;
        if (is_netcdf4)
        {
            flags |= NC_NETCDF4;
        }

        handle_error(nc_create(file_name.c_str(), flags, &ncid));

        handle_error(nc_def_dim(ncid, "x", NX, &x_dimid));
        handle_error(nc_def_dim(ncid, "y", NY, &y_dimid));
        handle_error(nc_def_dim(ncid, "z", NZ, &z_dimid));
        handle_error(nc_def_dim(ncid, "t", NT, &t_dimid));

        const char *var_name = "data";
        int dimids[4] = {x_dimid, y_dimid, z_dimid, t_dimid};
        handle_error(nc_def_var(ncid, var_name, NC_DOUBLE, 4, dimids, &varid));

        handle_error(nc_enddef(ncid));

        for (size_t j = 0; j < NX; j += chunk_size)
        {
            size_t start[4] = {j, 0, 0, 0};
            size_t count[4] = {min(static_cast<size_t>(chunk_size), static_cast<size_t>(NX) - j), static_cast<size_t>(NY), static_cast<size_t>(NZ), static_cast<size_t>(NT)};

            vector<double> data(count[0] * NY * NZ * NT, rank);

            handle_error(nc_put_vara_double(ncid, varid, start, count, data.data()));
        }

        handle_error(nc_close(ncid));
    }
}

void NetCDF_NtoN_read_4D(int rank, int size, const string &filename, bool is_netcdf4, const vector<int> &grid_dims_size, int file_count, int chunk_size)
{
    int ncid, varid;
    int NX = grid_dims_size[0], NY = grid_dims_size[1], NZ = grid_dims_size[2], NT = grid_dims_size[3];

    for (int i = 0; i < file_count; i++)
    {
        string file_name = filename + to_string(rank) + "_" + to_string(i) + ".nc";

        int flags = NC_NOWRITE;
        if (is_netcdf4)
        {
            flags |= NC_NETCDF4;
        }

        handle_error(nc_open(file_name.c_str(), flags, &ncid));

        handle_error(nc_inq_varid(ncid, "data", &varid)); // Get variable ID

        vector<double> data(NX * NY * NZ * NT);

        for (size_t j = 0; j < NX; j += chunk_size)
        {
            size_t start[4] = {j, 0, 0, 0};
            size_t count[4] = {min(static_cast<size_t>(chunk_size), static_cast<size_t>(NX) - j), static_cast<size_t>(NY), static_cast<size_t>(NZ), static_cast<size_t>(NT)};

            handle_error(nc_get_vara_double(ncid, varid, start, count, data.data()));
        }

        handle_error(nc_close(ncid));
    }
}

void delete_files(int rank, const string &filename, int file_count)
{
    for (int i = 0; i < file_count; i++)
    {
        string file_name = filename + to_string(rank) + "_" + to_string(i) + ".nc";
        remove(file_name.c_str());
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Default pattern：N-N
    string pattern = "N-N";
    // Default op_type
    char op_type = 'w';
    // Default file count: 1
    int file_count = 1;
    // Default file format
    bool is_netcdf4 = false;
    // Default dim
    int dimids = 3;
    // Default grid_dims_sizes
    vector<int> grid_dims_size;
    // Default step
    int step = 1;
    // Default filename
    string filename = "netcdf";
    // Default keep files
    bool keep_files = false;
    // Default iterations
    int iterations = 1;
    // Default sleep
    int sleep_seconds = 0;

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-p") == 0)
        {
            if (strcmp(argv[i + 1], "N-1") == 0)
            {
                pattern = "N-1";
            }
        }
        else if (strcmp(argv[i], "-n") == 0)
        {
            file_count = atoi(argv[i + 1]);
        }
        else if (strcmp(argv[i], "-f") == 0)
        {
            filename = argv[i + 1];
        }
        else if (strcmp(argv[i], "-r") == 0)
        {
            op_type = 'r';
        }
        else if (strcmp(argv[i], "-nc4") == 0)
        {
            is_netcdf4 = true;
        }
        else if (strcmp(argv[i], "-d") == 0)
        {
            dimids = atoi(argv[i + 1]);
            for (int j = 1; j <= dimids; j++)
                grid_dims_size.push_back(atoi(argv[i + 1 + j]));
        }
        else if (strcmp(argv[i], "-s") == 0)
        {
            step = atoi(argv[i + 1]);
        }
        else if (strcmp(argv[i], "-k") == 0)
        {
            keep_files = true;
        }
        else if (strcmp(argv[i], "-i") == 0)
        {
            iterations = atoi(argv[i + 1]);
        }
        else if (strcmp(argv[i], "-g") == 0)
        {
            sleep_seconds = atoi(argv[i + 1]);
        }
    }

    for (int iter = 0; iter < iterations; iter++)
    {
        // Delete existing files if not keeping them
        if (!keep_files)
        {
            delete_files(rank, filename, file_count);
        }

	std::cout << "iterations: " << iter << std::endl;

        if (op_type == 'w')
        {
            if (pattern == "N-N")
            {
                if (dimids == 2)
                {
                    NetCDF_NtoN_write_2D(rank, size, filename, is_netcdf4, grid_dims_size, file_count, step);
                }
                else if (dimids == 3)
                {
                    NetCDF_NtoN_write_3D(rank, size, filename, is_netcdf4, grid_dims_size, file_count, step);
                }
                else if (dimids == 4)
                {
                    NetCDF_NtoN_write_4D(rank, size, filename, is_netcdf4, grid_dims_size, file_count, step);
                }
            }
        }
        else if (op_type == 'r')
        {
            if (pattern == "N-N")
            {
                if (dimids == 2)
                {
                    NetCDF_NtoN_read_2D(rank, size, filename, is_netcdf4, grid_dims_size, file_count, step);
                }
                else if (dimids == 3)
                {
                    NetCDF_NtoN_read_3D(rank, size, filename, is_netcdf4, grid_dims_size, file_count, step);
                }
                else if (dimids == 4)
                {
                    NetCDF_NtoN_read_4D(rank, size, filename, is_netcdf4, grid_dims_size, file_count, step);
                }
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        if (sleep_seconds > 0)
        {
            sleep(sleep_seconds);
        }
    }

    MPI_Finalize();

    return 0;
}

