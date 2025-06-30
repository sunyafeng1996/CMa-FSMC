import time, gc
import torch
from thop import profile

def get_flops_paras(model, device, HW = 224):
    model = model.to(device)
    x = torch.randn(1, 3, HW, HW).to(device)
    flops, paras = profile(model, inputs=(x,), verbose=False)
    return flops, paras

# def get_latency(model, device, HW = 224):
#     input_tensor = torch.randn(8, 3, HW, HW).to(device)
#     model = model.to(device).eval()
#     with torch.no_grad():
#         for _ in range(1000):
#             _ = model(input_tensor)
#     num_runs = 5000
#     start_time = time.time()
#     with torch.no_grad():
#         for _ in range(num_runs):
#             _ = model(input_tensor)
#     if device == 'cuda':
#         torch.cuda.synchronize()
#     elapsed_time = time.time() - start_time
#     avg_time = elapsed_time# / num_runs
#     return avg_time

def get_latency(model, device, HW = 224, amp=False, test_time=500):
    data = torch.randn(64, 3, HW, HW).to(device)
    model = model.to(device).eval()
    st = time.time()
    print('=> testing latency. Please wait.')
    with torch.no_grad():
        output = model(data)
    if amp:
        start_timer()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                for i in range(test_time):
                    output = model(data)
        total_time = end_timer()
        each_time = total_time / test_time
    else:
        start_timer()
        with torch.no_grad():
            for i in range(test_time):
                output = model(data)
        total_time = end_timer()
        each_time = total_time / test_time
    et = time.time()
    print('=> testing latency finshed. cost {:.4}s'.format(et-st))
    return each_time

def start_timer():
    global start_time
    gc.collect()
    torch.cuda.empty_cache()
    #torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start_time = time.time()

def end_timer():
    torch.cuda.synchronize()
    end_time = time.time()
    return end_time - start_time

def end_timer_and_print(local_msg):
    torch.cuda.synchronize()
    end_time = time.time()
    return end_time - start_time