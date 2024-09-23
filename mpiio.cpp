#include <mpi.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <cstring>
#include <sys/stat.h>

using namespace std;

// 将字符串表示的大小转换为字节数
long long parse_size_mpiio(const char* str_size){
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

void MPIIO_NtoN_write(int rank, int size, string filename, long long total_size, long long request_size, int file_count, bool continuous_op, bool keep_files)
{
    MPI_File fh;
    MPI_Status status;

    string filename_t = filename;

    for(int i = 0; i < file_count; i++){
        if(size != 1){
            filename_t = filename + to_string(rank) + "_" + to_string(i);
        }

        int open_status = MPI_File_open(MPI_COMM_SELF, filename_t.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
        if (open_status != MPI_SUCCESS) {
            cout << "无法打开文件: " << filename_t << endl;
            MPI_Abort(MPI_COMM_WORLD, open_status);
        }
        
        char *buffer = new char[request_size];
        memset(buffer, 'a' + rank, request_size);
        long long written = 0;
        while (written < total_size){
            MPI_File_write(fh , buffer , request_size , MPI_CHAR , &status);
            written += request_size;
        }

        delete[] buffer;
        MPI_File_close(&fh);
        
        if(!keep_files) {
            remove(filename_t.c_str());
        }
    }
}

void MPIIO_NtoN_read(int rank, int size, string filename, long long total_size, long long request_size, int file_count, bool continuous_op, bool keep_files)
{
    MPI_File fh;
    MPI_Status status;

    string filename_t = filename;

    for(int i = 0; i < file_count; i++){
        if(size != 1){
            filename_t = filename + to_string(rank) + "_" + to_string(i);
        }

        int open_status = MPI_File_open(MPI_COMM_SELF, filename_t.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
        if (open_status != MPI_SUCCESS) {
            cout << "无法打开文件: " << filename_t << endl;
            MPI_Abort(MPI_COMM_WORLD, open_status);
        }

        char *buffer = new char[request_size];
        memset(buffer, 'a' + rank, request_size);
        long long readsize = 0;
        while (readsize < total_size){
            MPI_File_read(fh , buffer , request_size , MPI_CHAR , &status);
            readsize += request_size;
        }

        delete[] buffer;
        MPI_File_close(&fh);

        if(!keep_files) {
            remove(filename_t.c_str());
        }
    }
}

void MPIIO_Nto1_write(int rank, int size, string filename, long long total_size, long long request_size, int file_count, bool continuous_op, bool keep_files)
{
    MPI_File fh;
    MPI_Status status;

    int open_status = MPI_File_open(MPI_COMM_WORLD , filename.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    if (open_status != MPI_SUCCESS) {
        cout << "无法打开文件: " << filename << endl;
        MPI_Abort(MPI_COMM_WORLD, open_status);
    }

    char *buffer = new char[request_size];
    memset(buffer, 'a' + rank, request_size);

    MPI_Offset offset = continuous_op ? rank * total_size : rank * request_size;

    for(long long i = 0; i < total_size / request_size; i++){
        MPI_File_write_at_all(fh, offset, buffer, request_size, MPI_CHAR, &status);
        offset += continuous_op ? request_size : request_size * size;
    }

    delete[] buffer;
    MPI_File_close(&fh);
    
    if(!keep_files) {
        remove(filename.c_str());
    }
}

void MPIIO_Nto1_read(int rank, int size, string filename, long long total_size, long long request_size, int file_count, bool continuous_op, bool keep_files)
{
    MPI_File fh;
    MPI_Status status;

    int open_status = MPI_File_open(MPI_COMM_WORLD , filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    if (open_status != MPI_SUCCESS) {
        cout << "无法打开文件: " << filename << endl;
        MPI_Abort(MPI_COMM_WORLD, open_status);
    }

    char *buffer = new char[request_size];
    memset(buffer, 'a' + rank, request_size);

    MPI_Offset offset = continuous_op ? rank * total_size : rank * request_size;

    for(long long i = 0; i < total_size / request_size; i++){
        MPI_File_read_at_all(fh, offset, buffer, request_size, MPI_CHAR, &status);
        offset += continuous_op ? request_size : request_size * size;
    }

    delete[] buffer;
    MPI_File_close(&fh);
    
    if(!keep_files) {
        remove(filename.c_str());
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    struct stat st;
    stat("/datafiles/a", &st);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 默认参数
    string pattern = "N-N";
    char op_type = 'w'; 
    long long request_size = 512 * 1024; 
    long long total_size = 512 * 1024 * 1024; 
    int file_count = 1; 
    bool continuous_op = true; 
    string filename = "mpiiodefault"; 
    bool keep_files = false;
    int iterations = 1;
    int sleep_seconds = 0;
    
    // 解析命令行参数
    for (int i = 1; i < argc; i++){
        if(strcmp(argv[i], "-p") == 0 && i + 1 < argc){
            if(strcmp(argv[i + 1], "N-1") == 0){
                pattern = "N-1";
            }
        }
        else if(strcmp(argv[i], "-b") == 0 && i + 1 < argc){
            total_size = parse_size_mpiio(argv[i + 1]);
        }
        else if(strcmp(argv[i], "-t") == 0 && i + 1 < argc){
            request_size = parse_size_mpiio(argv[i + 1]);
        }
        else if(strcmp(argv[i], "-n") == 0 && i + 1 < argc){
            file_count = atoi(argv[i + 1]);
        }
        else if(strcmp(argv[i], "-f") == 0 && i + 1 < argc){
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
        else if(strcmp(argv[i], "-i") == 0 && i + 1 < argc){
            iterations = atoi(argv[i + 1]);
        }
        else if(strcmp(argv[i], "-g") == 0 && i + 1 < argc){
            sleep_seconds = atoi(argv[i + 1]);
        }
    }

    // 执行读写操作
    for(int i = 0; i < iterations; i++){
    
	std::cout << "mpiio_iteration: " << i << std::endl;
        if(op_type == 'w'){
            if(pattern == "N-N"){
                MPIIO_NtoN_write(rank, size, filename, total_size, request_size, file_count, continuous_op, keep_files);
            }
            if(pattern == "N-1"){
                MPIIO_Nto1_write(rank, size, filename, total_size, request_size, file_count, continuous_op, keep_files);
            }
        }
        if(op_type == 'r'){
            if(pattern == "N-N"){
                MPIIO_NtoN_read(rank, size, filename, total_size, request_size, file_count, continuous_op, keep_files);
            }
            if(pattern == "N-1"){
                MPIIO_Nto1_read(rank, size, filename, total_size, request_size, file_count, continuous_op, keep_files);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        sleep(sleep_seconds);
    }
    
    MPI_Finalize();

    return 0;
}

