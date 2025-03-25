import sys
import logging
import argparse
import os
import time
import psutil
import numpy as np

def setup_logging():
    # Create log directory if it doesn't exist
    log_dir = "/tmp"
    log_file = os.path.join(log_dir, "allocate_memory.log")
    
    # Configure logging format
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("Logging initialized. Log file: %s", log_file)

def parse_args():
    parser = argparse.ArgumentParser(description='Allocate memory and monitor usage')
    parser.add_argument('--memory-mb', type=int, default=1024,
                      help='Total memory to allocate in MB')
    parser.add_argument('--hold-time', type=int, default=3600,
                      help='Time to hold the memory in seconds')
    parser.add_argument('--step-size-mb', type=int, default=100,
                      help='Memory to allocate in each step in MB')
    parser.add_argument('--step-interval', type=float, default=0.5,
                      help='Interval between steps in seconds')
    return parser.parse_args()

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

def main():
    args = parse_args()
    setup_logging()
    
    logging.info("Starting memory allocation with parameters: %s", args)
    
    # 计算每一步需要分配的数组大小
    step_size_bytes = args.step_size_mb * 1024 * 1024  # Convert MB to bytes
    array_size = int(step_size_bytes / 8)  # 8 bytes per float64
    total_steps = int(args.memory_mb / args.step_size_mb)
    
    # 保存所有分配的数组，防止被垃圾回收
    arrays = []
    
    try:
        # 逐步分配内存
        for step in range(total_steps):
            initial_memory = get_memory_usage()
            
            # 分配内存
            array = np.random.rand(array_size)
            arrays.append(array)
            
            current_memory = get_memory_usage()
            allocated_memory = current_memory - initial_memory
            total_allocated = current_memory
            
            logging.info(f"Step {step + 1}/{total_steps}: "
                        f"Allocated {allocated_memory:.2f}MB, "
                        f"Total memory usage: {total_allocated:.2f}MB "
                        f"({total_allocated/1024:.2f}GB)")
            
            time.sleep(args.step_interval)
        
        total_memory = get_memory_usage()
        logging.info(f"Memory allocation completed. "
                    f"Total memory usage: {total_memory:.2f}MB "
                    f"({total_memory/1024:.2f}GB)")
        
        # 保持内存占用
        logging.info(f"Holding memory for {args.hold_time} seconds...")
        time.sleep(args.hold_time)
        
    except KeyboardInterrupt:
        logging.info("Received keyboard interrupt, cleaning up...")
    except Exception as e:
        logging.error(f"Error occurred: {e}")
    finally:
        # 清理内存
        arrays.clear()
        final_memory = get_memory_usage()
        logging.info(f"Final memory usage: {final_memory:.2f}MB "
                    f"({final_memory/1024:.2f}GB)")

if __name__ == "__main__":
    main() 