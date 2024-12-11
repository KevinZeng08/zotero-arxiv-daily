def select_gpu_with_max_free_memory(num_gpus: int = 1):
    import pynvml
    pynvml.nvmlInit()
    try:
        device_count = pynvml.nvmlDeviceGetCount()
        free_memory_list = []
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            free_memory = info.free
            free_memory_list.append(free_memory)

        sorted_free_memory_list = sorted(zip(free_memory_list, range(device_count)), reverse=True)
        selected_gpus = [i for _, i in sorted_free_memory_list[:num_gpus]]

        return selected_gpus
    finally:
        pynvml.nvmlShutdown()