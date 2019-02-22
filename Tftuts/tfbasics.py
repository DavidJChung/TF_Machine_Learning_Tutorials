import tensorflow as tf

x1 = tf.constant(5)             #creates tensor constants
x2 = tf.constant(6)


result = tf.multiply(x1,x2)

print(result)

with tf.Session() as sess:              # dont have to close session
    output = sess.run(result)            # session con only be accessed within this block
    print(output)

                                        #output becomes python varibale and can be accessed outside the session

# sess = tf.Session()                       # runs tensor model
# print(sess.run(result))
# sess.close()