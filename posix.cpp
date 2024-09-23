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

using namespace std;

long long parse_size_posixio(const char* str_size){
    long long multiplier = 1;
    long long value = strtoll(str_size, nullptr, 10);
    char unit = str_size[strlen(str_size) - 1];
    
    switch (unit){
        case 'k':
        case 'K':
            multiplier = 1024;
            break;
        case 'm':
        case 'M':
            multiplier = 1024 * 1024;
            break;
        case 'g':
        case 'G':
            multiplier = 1024 * 1024 * 1024;
            break;
        default:
            cout << "单位不正确" << endl;
            break;
    }

    return value * multiplier;
}

void POSIXIO_write(int rank, int size, string filename, long long total_size, long long request_size, int file_count, bool continuous_op, bool keep_files)
{
    string filename_t;
    for(int i = 0; i < file_count; i++){
        if(size != 1){
            filename_t = filename + to_string(rank) + "_" + to_string(i);
        } else {
            filename_t = filename;
        }
        
        int fd = open(filename_t.c_str(), O_WRONLY | O_CREAT, S_IRUSR | S_IWUSR);
        if (fd < 0) {
            cerr << "无法打开文件: " << filename_t << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        char *buffer = new char[request_size];
        memset(buffer, 'a' + rank, request_size);
        long long written = 0;
        while (written < total_size){
            ssize_t ret = write(fd, buffer, request_size);
            if (ret != request_size) {
                cerr << "写入错误: " << filename_t << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            written += request_size;
        }

        delete[] buffer;
        close(fd);
        if(!keep_files)
            remove(filename_t.c_str());
    }
}

void POSIXIO_read(int rank, int size, string filename, long long total_size, long long request_size, int file_count, bool continuous_op, bool keep_files)
{
    string filename_t;
    for(int i = 0; i < file_count; i++){
        if(size != 1){
            filename_t = filename + to_string(rank) + "_" + to_string(i);
        } else {
            filename_t = filename;
        }
        
        int fd = open(filename_t.c_str(), O_RDONLY, S_IRUSR | S_IWUSR);
        if (fd < 0) {
            cerr << "无法打开文件: " << filename_t << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        char *buffer = new char[request_size];
        memset(buffer, 'a' + rank, request_size);
        long long readsize = 0;
        while (readsize < total_size){
            ssize_t ret = read(fd, buffer, request_size);
            if (ret != request_size) {
                cerr << "读取错误: " << filename_t << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            readsize += request_size;
        }

        delete[] buffer;
        close(fd);
        if(!keep_files)
            remove(filename_t.c_str());
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Default op_type
    char op_type = 'w'; 
    // Default request size: 512kb
    long long request_size = 512 * 1024; 
    // Default total data size: 512mb
    long long total_size = 512 * 1024 * 1024; 
    // Default file count: 1
    int file_count = 1; 
    // Default continuous: true
    bool continuous_op = true; 
    // Default filename
    string filename = "posixio"; 
    // Default keep
    bool keep_files = false;
    // Default iterations
    int iterations = 1;
    // Default sleep
    int sleep_seconds = 0;
    
    // parse command line
    for (int i = 1; i < argc; i++){
        if(strcmp(argv[i], "-b") == 0){
            total_size = parse_size_posixio(argv[i + 1]);
        }
        else if(strcmp(argv[i], "-t") == 0){
            request_size = parse_size_posixio(argv[i + 1]);
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
        else if(strcmp(argv[i], "-z") == 0){
            continuous_op = false;
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
	std::cout << "posix_iterations: " << i << std::endl;
        if (op_type == 'w'){
            POSIXIO_write(rank, size, filename, total_size, request_size, file_count, continuous_op, keep_files);
        }
        if(op_type == 'r'){
            POSIXIO_read(rank, size, filename, total_size, request_size, file_count, continuous_op, keep_files);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        sleep(sleep_seconds);
    }

    MPI_Finalize();
    return 0;
}

