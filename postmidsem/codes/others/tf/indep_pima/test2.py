import tensorflow as tf
import numpy as np
def main():
	y=tf.Variable(np.zeros((4,1)),'y')
	savo=tf.train.Saver(var_list=[y])
	n=y.assign(np.arange(4).reshape(4,1))
	with tf.Session() as sess:
		sess.run(n)
		print("saving checkpoint")
		save_path = savo.save(sess, "/home/placements2018/forgit/neuro-evolution/05/state/tf/test/model.ckpt")
		print(y.eval())
	func(y)
def func(y):
	savo1=tf.train.Saver(var_list=[y])
	with tf.Session() as sess:
		savo1.restore(sess, "/home/placements2018/forgit/neuro-evolution/05/state/tf/test/model.ckpt")
		print("func",y.eval())
	with tf.Session() as sess:
		print(y.eval())

main()