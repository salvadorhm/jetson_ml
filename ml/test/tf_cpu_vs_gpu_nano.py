import tensorflow as tf
import time
import matplotlib.pyplot as plt

cpu_times = []

tf.debugging.set_log_device_placement(True)
sizes = [1, 10, 100, 500, 1000, 2000]
for size in sizes:
    start = time.time()
    with tf.device('/CPU:0'):
        a = tf.constant(tf.random.normal((size, size)))
        b = tf.constant(tf.random.normal((size, size)))

        # Run on the CPU
        c = tf.matmul(a, b)
        cpu_times.append(time.time() - start)
        print('cpu time took: {0:.4f}'.format(time.time() - start))
        print(c)


gpu_times = []
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

tf.debugging.set_log_device_placement(True)
sizes = [1, 10, 100, 500, 1000, 2000]
for size in sizes:
    start = time.time()
    with tf.device('/device:GPU:0'):
        a = tf.constant(tf.random.normal((size, size)))
        b = tf.constant(tf.random.normal((size, size)))

        # Run on the GPU
        c = tf.matmul(a, b)
        gpu_times.append(time.time() - start)
        print('cpu time took: {0:.4f}'.format(time.time() - start))
        print(c)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(sizes, gpu_times, label='GPU')
ax.plot(sizes, cpu_times, label='CPU')
plt.xlabel('MATRIX SIZE')
plt.ylabel('TIME (sec)')
plt.legend()
plt.title("CPU vs GPU Nvidia Jetson Nano 2GB")
plt.savefig("cpu_gpu_jetson.png")

print("PLOT")
