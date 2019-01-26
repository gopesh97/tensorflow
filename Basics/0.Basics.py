import tensorflow as tf

t1 = tf.constant(3.0)
t2 = tf.constant(2.0)
tv = tf.Variable(5.0)
tp = tf.placeholder(dtype=tf.float32)
result = t1 + t2
res2 = tv + result
res3 = tp + tv

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print(sess.run(res2))
	print(sess.run(res3,{tp:50.0}))

	fw = tf.summary.FileWriter('tf_graph',sess.graph)