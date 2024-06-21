import numpy as np
import matplotlib.pyplot as plt
import random
import time

minv = 1
maxv = 200
 
def test_time_sort_list(n):
    data = [random.randint(minv, maxv) for i in range(n)]
    start = time.time()
    data.sort()
    end = time.time() - start
    return end
 
def test_time_sort_numpy(n):
    
    data_numpy = np.random.randint(minv, maxv, n);
    start = time.time()
    data_numpy.sort()
    end = time.time() - start
    return end
 
 
start_program = time.time()
arr_times_numpy = []
arr_times_list = []

arr_iters = []
step = 100000
start_iter = 10
end_iter = 3000000
adf = 0
for i in range(start_iter, end_iter, step):
    
    print(i)
    arr_iters.append(i)
    arr_times_list.append(test_time_sort_list(i))
    arr_times_numpy.append(test_time_sort_numpy(i))
    
print("Draw")
plt.plot(arr_iters, arr_times_list)
plt.plot(arr_iters, arr_times_numpy, 'r')

plt.xlabel("Количество итераций(шаг{})".format(step))
plt.ylabel("Время сортировки")
plt.legend(["List", "Numpy"])
plt.show()
print("End program: ", time.time() - start_program)
print("Numpy: ", arr_times_numpy[29])
print("List: ", arr_times_list[29])

 
 