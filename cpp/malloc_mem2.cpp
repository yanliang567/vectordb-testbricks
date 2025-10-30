#include <iostream>
#include <cstdlib>
#include <sys/mman.h>   // mmap, mlock
#include <unistd.h>     // getpagesize
#include <cstring>      // memset
#include <vector>

void* allocate_and_touch(size_t size_in_bytes) {
    void* mem = mmap(nullptr, size_in_bytes, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mem == MAP_FAILED) {
        perror("mmap failed");
        return nullptr;
    }

	 // 锁定内存，防止被 swap
    if (mlock(mem, size_in_bytes) != 0) {
        perror("mlock failed");
        munmap(mem, size_in_bytes);
    }

    if (mlock(mem, size_in_bytes) != 0) {
        perror("mlock failed");
        munmap(mem, size_in_bytes);
        return nullptr;
    }

    if (madvise(mem, size_in_bytes, MADV_WILLNEED) != 0) {
        perror("madvise failed");
    }

    // 每页写一次，强制分配物理内存
    int page_size = getpagesize();
    char* p = static_cast<char*>(mem);
    for (size_t offset = 0; offset < size_in_bytes; offset += page_size) {
        p[offset] = 1;
    }

    return mem;
}

int main(int argc, char* argv[]) {
    // run example: g++ 1.cpp && ./a.out 1000 10

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <total_memory_in_MB> [chunk_size_in_MB]" << std::endl;
        return 1;
    }

    int total_memory_mb = std::atoi(argv[1]);
    int chunk_size_mb = (argc >= 3) ? std::atoi(argv[2]) : 128; // 默认每次分128MB

    if (total_memory_mb <= 0 || chunk_size_mb <= 0) {
        std::cerr << "Invalid memory size or chunk size." << std::endl;
        return 1;
    }

    size_t total_memory = static_cast<size_t>(total_memory_mb) * 1024 * 1024;
    size_t chunk_size = static_cast<size_t>(chunk_size_mb) * 1024 * 1024;

    std::cout << "Allocating " << total_memory_mb << "MB total, in chunks of " 
              << chunk_size_mb << "MB each..." << std::endl;

    std::vector<void*> allocations;
    size_t allocated = 0;

    while (allocated < total_memory) {
        size_t size = std::min(chunk_size, total_memory - allocated);
        void* mem = allocate_and_touch(size);
        if (!mem) {
            std::cerr << "Allocation failed after " << allocated / (1024*1024) << " MB" << std::endl;
            break;
        }
        allocations.push_back(mem);
        allocated += size;
        std::cout << "Allocated and touched " << allocated / (1024*1024) << " MB..." << std::endl;
    }

    std::cout << "All allocations done. Press ENTER to exit and release memory..." << std::endl;
    std::cin.get();

    for (auto mem : allocations) {
        munmap(mem, chunk_size);
    }

    return 0;
}
