/*
POSIXIO模拟程序
可调参数:
    1.IO模式: N-N
    2.读 or 写
    3.读写并行度
    4.文件数量、文件大小
    5.每次I/O请求大小
    6.读写连续性 (可选)
    7.指定文件名 (可选)
    8.迭代次数
    9.间隔时间
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

using namespace std;

long long parse_size_posixio(const char* str_size){
    long long multiplier = 1;
    long long value = strtoll(str_size, nullptr, 10);
    char unit = str_size[strlen(str_size) - 1];
    
    switch (unit){
        case 'k':
        case 'K':{
            multiplier = 1024;
            break;
        }
        case 'm':
        case 'M':{
            multiplier = 1024 * 1024;
            break;
        }
        case 'g':
        case 'G':{
            multiplier = 1024 * 1024 * 1024;
            break;
        }
        default:
            cout << "单位不正确" << endl;
            break;
    }

    return value * multiplier;
}

void POSIXIO_write(int rank, int size, string filename, long long total_size, long long request_size, int file_count, bool continuous_op, bool keep_files)
{
    string filename_t = filename;

    for(int i = 0; i < file_count; i++){
        if(continuous_op){
            if(size != 1){
                filename_t = filename + to_string(rank) + "_" + to_string(i);
            }
            
            int fd = open(filename_t.c_str(), O_WRONLY | O_CREAT, S_IRUSR | S_IWUSR);
            char *buffer = new char[request_size];
            memset(buffer, 'a' + rank, request_size);
            long long written = 0;
            while (written < total_size){
                write(fd, buffer, request_size);
                written += request_size;
            }

            delete[] buffer;
            close(fd);
        }
        else{
            if(size != 1){
                filename_t = filename + to_string(rank) + "_" + to_string(i);
            }
            
            int fd = open(filename_t.c_str(), O_WRONLY | O_CREAT, S_IRUSR | S_IWUSR);
            char *buffer = new char[request_size];
            memset(buffer, 'a' + rank, request_size);

            MPI_Offset offset = 0;
            for(long long written = 0; written < total_size; written += request_size){
                
                lseek(fd, offset, SEEK_SET);
                write(fd, buffer, request_size);

                if(offset >= total_size) {
                    offset = request_size;
                }
                else {
                    offset += request_size * 2;
                }
            }
            delete[] buffer;
            close(fd);
            if(!keep_files)
                remove(filename_t.c_str());
        }
    }
}

void POSIXIO_read(int rank, int size, string filename, long long total_size, long long request_size, int file_count, bool continuous_op, bool keep_files)
{
    string filename_t = filename;

    for(int i = 0; i < file_count; i++){
        if(continuous_op){
            if(size != 1){
                filename_t = filename + to_string(rank) + "_" + to_string(i);
            }
            
            int fd = open(filename_t.c_str(), O_RDONLY, S_IRUSR | S_IWUSR);
            if(fd < 0){
                printf("打开文件%s失败\n", filename_t.c_str());
                return ;
            }


            char *buffer = new char[request_size];
            memset(buffer, 'a' + rank, request_size);
            long long written = 0;
            while (written < total_size){
                read(fd, buffer, request_size);
                written += request_size;
            }

            delete[] buffer;
            close(fd);
        }
        else{
            if(size != 1){
                filename_t = filename + to_string(rank) + "_" + to_string(i);
            }
            
            int fd = open(filename_t.c_str(), O_RDONLY, S_IRUSR | S_IWUSR);
            char *buffer = new char[request_size];
            memset(buffer, 'a' + rank, request_size);

            MPI_Offset offset = 0;
            for(long long written = 0; written < total_size; written += request_size){
                
                lseek(fd, offset, SEEK_SET);
                read(fd, buffer, request_size);

                if(offset >= total_size) {
                    offset = request_size;
                }
                else {
                    offset += request_size * 2;
                }
            }
            delete[] buffer;
            close(fd);
            if(!keep_files)
                remove(filename_t.c_str());
        }
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    //Default op_type
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
    
    //parse command line
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

        if (op_type == 'w'){
            POSIXIO_write(rank, size, filename, total_size, request_size, file_count, continuous_op, keep_files);
        }
        if(op_type == 'r'){
            POSIXIO_read(rank, size, filename, total_size, request_size, file_count, continuous_op, keep_files);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        sleep(sleep_seconds);

    }
    return 0;
}