

import numpy as np
import tensorflow as tf


def test():
        def func(z,y):
            dum = tf.constant(0.5,dtype=tf.float32)
            minusone = tf.constant(-1,dtype=tf.int32)
            #dum = tf.constant(0.5, dtype=tf.float32)  # dum for dummy
            dadum = tf.constant(-1, dtype=tf.int32)  # dum-dadum-dadum mast h
            p = tf.scan(fn=lambda last,current : [last[0]+1, current], elems=y, initializer=[dadum, dadum])
            l = tf.transpose(tf.stack([p[0], p[1]]))
            one = tf.constant(1,dtype=tf.int32)
            r = tf.scan(lambda last, current:last+1, elems=y, initializer=minusone)
            w = tf.scan(lambda last, current: [z[current][0], 1 - z[current][0]],
                                       elems=r, initializer=[dum, dum])
            k = tf.transpose(tf.stack([w[0], w[1]]))

            return k,l

        yar = np.arange(4)
        zar = np.random.random((4,1))
        z = tf.placeholder(shape = (None, 1), dtype = tf.float32)

        y = tf.placeholder(dtype=tf.int32, shape = (None,))
        with tf.Session() as sess:
            l = sess.run(func(z,y), feed_dict = {z:zar, y :yar})
        print(l)


            #self.P_Y_GIVEN_X = tf.scan(lambda last, current: [self.p_y_given_x[current], 1 - self.p_y_given_x[current]], elems = r, initializer = [dum,dum])
            #dadum = tf.constant(-1, dtype=tf.int32)  # dum-dadum-dadum mast h
            #q = tf.scan(fn=func, elems=y, initializer=[dadum, dadum])
            #z = tf.transpose(tf.stack([q[0], q[1]]))
            #self.P_Y_GIVEN_X = tf.scan(lambda last, current: [self.p_y_given_x[current], 1 - self.p_y_given_x[current]],
            #                           elems=r, initializer=[dum, dum])

            # print("hello---------------------------")
            #w = tf.scan(lambda last, current: tf.log(self.P_Y_GIVEN_X[current[0]][current[1]]), elems=z,
            #            initializer=dum)
            # print(-tf.reduce_mean(w))
            #return -tf.reduce_mean(w)
test()