import tensorflow as tf
# matrix = 4, 3, 2, 5

# cari nilai di batch 3 fitur 2 dan 4
# cari elemen terbesar

c = tf.random.uniform([4, 3, 2, 5])
print(c[2, 1, 3, :])
