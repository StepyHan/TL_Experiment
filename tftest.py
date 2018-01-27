import tensorflow as tf

a = tf.constant([0,0,1,0,1,0], shape=[2,3])
ns = a.get_shape()[0]
es = tf.where

with tf.Session() as sess:
    # tf.boolean_mask(es, tf.equal(es, 0)) = 1
    print(es.eval())
    print(tf.equal(es, 0).eval())
    print()
    print(a.eval())
    print(tf.argmax(a, axis=0).eval())
    print(a.get_shape()[0])

xs = tf.random_uniform((4, 6), dtype=tf.float32)
xt = tf.random_uniform((4, 6), dtype=tf.float32)
BATCH_SIZE = 4
X = tf.concat([tf.transpose(xs), tf.transpose(xt)], 1)
yss = tf.constant([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 1, 0, 0]])
ytt = tf.constant([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
yss = tf.argmax(yss, axis=1)
ytt = tf.argmax(ytt, axis=1)

ns = BATCH_SIZE
nt = BATCH_SIZE
N = 0
z = tf.constant(value=0, shape=[BATCH_SIZE, 1], dtype=tf.float32)
s = tf.constant(value=1 / ns, shape=[ns, 1])
t = tf.constant(value=1 / nt, shape=[nt, 1])
for c in range(4):
    es = tf.where(tf.equal(yss, c), s, z)
    et = tf.where(tf.equal(ytt, c), t, z)
    e = tf.concat([es, et], 0)
    N = N + tf.matmul(e, tf.transpose(e))
mmd_con = tf.trace(tf.matmul(tf.matmul(X, N), tf.transpose(X)))

c = 1
ns = 6
a = tf.boolean_mask(yss, tf.equal(yss, c))
b = tf.shape(a)[0]
s_con = tf.cast(tf.Variable(1/b, trainable=False), tf.float32)
c = tf.reshape(s_con, [1, 1])
multi = tf.tile(input=c, multiples=[ns, 1])
a = tf.Variable(0, trainable=False, name='temp_')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(a.eval())
    tf.global_variables_initializer().run()
    print(c.eval())
    print(multi.eval())
    print(s_con.eval())

s_shape = tf.boolean_mask(yss, tf.equal(yss, c)).get_shape().as_list()[0]
t_shape = tf.boolean_mask(ytt, tf.equal(ytt, c)).get_shape().as_list()[0]


import tensorflow as tf
xs = tf.random_uniform((4, 6), dtype=tf.float32)
xt = tf.random_uniform((4, 6), dtype=tf.float32)
yss = tf.constant([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 1, 0, 0]])
ytt = tf.constant([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
yss = tf.argmax(yss, axis=1)
ytt = tf.argmax(ytt, axis=1)
N = 0



with tf.Session() as sess:
    for c in range(5):
        b = tf.boolean_mask(xs, tf.tile(tf.reshape(tensor=tf.equal(yss, c), shape=[-1, 1]), [1, tf.shape(xs)[1]]))
        xs_c = tf.reshape(tensor=b, shape=[-1, tf.shape(xs)[1]])  # TODO tile boolean mask to n_feature columns
        xt_c = tf.reshape(tf.boolean_mask(xt, tf.tile(tf.reshape(tf.equal(ytt, c), [-1, 1]), [1, tf.shape(xt)[1]])),
                          [-1, tf.shape(xt)[1]])

        a = tf.reduce_sum(tf.multiply((tf.reduce_mean(xs_c, 0) - tf.reduce_mean(xt_c, 0)),
                                      (tf.reduce_mean(xs_c, 0) - tf.reduce_mean(xt_c, 0))))
        a = tf.where(tf.is_nan(a), 0., a)
        print(a.eval())

