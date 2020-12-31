import os
import os.path
import tensorflow as tf

LOGDIR = "/tmp/tensorboard/"
LABELS = os.path.join(os.getcwd(), "labels-1024.tsv")
SPRITES = os.path.join(os.getcwd(), "sprite-1024.png")
mnist = tf.contrib.learn.datasets.mnist.read_data_sets(train_dir=LOGDIR + "data", one_hot=True)

if not (os.path.isfile(LABELS) and os.path.isfile(SPRITES)):
  print("LABELS veya SPRITES bulunamadı. Lütfen kurs dosyaları içerisinden indirin.")
  exit(1)


def conv_layer(input, size_in, size_out, name="conv"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([5, 5, size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
        act = tf.nn.relu(conv + b)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def fc_layer(input, size_in, size_out, name="fc"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        act = tf.matmul(input, w) + b
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act


def mnist_model(learning_rate, hparam):
    tf.reset_default_graph()
    sess = tf.Session()

    x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', x_image, 3)
    y = tf.placeholder(tf.float32, shape=[None, 10], name="labels")


    conv1 = conv_layer(x_image, 1, 32, "conv1")
    conv_out = conv_layer(conv1, 32, 64, "conv2")

    flattened = tf.reshape(conv_out, [-1, 7 * 7 * 64])

    fc1 = fc_layer(flattened, 7 * 7 * 64, 1024, "fc1")
    relu = tf.nn.relu(fc1)
    embedding_input = relu
    tf.summary.histogram("fc1/relu", relu)
    embedding_size = 1024
    logits = fc_layer(fc1, 1024, 10, "fc2")

    with tf.name_scope("xent"):
        xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y), name="xent")
        tf.summary.scalar("xent", xent)

    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(xent)

    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

    summ = tf.summary.merge_all()

    embedding = tf.Variable(tf.zeros([1024, embedding_size]), name="test_embedding")
    assignment = embedding.assign(embedding_input)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(LOGDIR + hparam)
    writer.add_graph(sess.graph)

    config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
    embedding_config = config.embeddings.add()
    embedding_config.tensor_name = embedding.name
    embedding_config.sprite.image_path = SPRITES
    embedding_config.metadata_path = LABELS
    embedding_config.sprite.single_image_dim.extend([28, 28])
    tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)

    for i in range(2001):
        batch = mnist.train.next_batch(100)
        if i % 5 == 0:
            [train_accuracy, s] = sess.run([accuracy, summ], feed_dict={x: batch[0], y: batch[1]})
            writer.add_summary(s, i)
        if i % 500 == 0:
            sess.run(assignment, feed_dict={x: mnist.test.images[:1024], y: mnist.test.labels[:1024]})
            saver.save(sess, os.path.join(LOGDIR, "model.ckpt"), i)
        sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})

def make_hparam_string(learning_rate):
    conv_param = "conv=2"
    fc_param = "fc=2"
    return "lr_%.0E,%s,%s" % (learning_rate, conv_param, fc_param)

def main():
    learning_rate = 5e-4
    hparam = make_hparam_string(learning_rate)
    print('Starting run for %s' % hparam)
    mnist_model(learning_rate, hparam)
    print('Done training!')
    print('Run `tensorboard --logdir=%s` to see the results.' % LOGDIR)


if __name__ == '__main__':
    main()
