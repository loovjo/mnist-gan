# Fix ssl
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import random
import tensorflow as tf
import numpy as np
import os

SAVE_PATH = os.path.join("saves", "local", "networks")
LOG_PATH = os.path.join("saves", "local", "log")

for f in SAVE_PATH, LOG_PATH:
    if not os.path.isdir(f):
        os.mkdir(f)

BATCH_SIZE = 300
NOISE_SIZE = 30

dataset = tf.keras.datasets.mnist
(TRAIN_IMAGES, _), (TEST_IMAGES, _) = dataset.load_data()


class GAN:
    def __init__(self, net_id, lr_gen=None, lr_disc=None):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)

        with self.graph.as_default():
            if lr_gen == None:
                self.lr_gen = tf.get_variable(dtype="float32", name="lr_gen", shape=[])
            else:
                self.lr_gen = tf.Variable(lr_gen, name="lr_gen")

            if lr_disc == None:
                self.lr_disc = tf.get_variable(dtype="float32", name="lr_disc", shape=[])
            else:
                self.lr_disc = tf.Variable(lr_disc, name="lr_disc")


            savefolder = os.path.join(SAVE_PATH, net_id)
            if not os.path.isdir(savefolder):
                os.mkdir(savefolder)

            self.savepath = os.path.join(savefolder, "save")
            logpath = os.path.join(LOG_PATH, net_id)

            self.fw = tf.summary.FileWriter(logpath)

            self.epoch = tf.Variable(0, name="epoch")

            # Generator def

            with tf.name_scope("generator"):
                self.gen_l1 = tf.layers.Dense(7 * 7)
                self.gen_l2 = tf.layers.Conv2DTranspose(
                    8, [4, 4], strides=(2, 2), padding="same")
                self.gen_l3 = tf.layers.Conv2DTranspose(
                    1, [6, 6], strides=(2, 2), padding="same")

            self.gen_layers = [self.gen_l1, self.gen_l2, self.gen_l3]

            # noise->gen

            with tf.name_scope("noise-gen"):
                self.gnoise = tf.placeholder(
                    dtype="float32", shape=[None, NOISE_SIZE], name="noise")
                self.ngen = self.gen(self.gnoise)

                tf.summary.image("generated",
                                 tf.reshape(self.ngen, [-1, 28, 28, 1]))

            # Discriminator def

            with tf.name_scope("discriminator"):
                self.disc_l1 = tf.layers.Conv2D(
                    8, [4, 4], strides=(2, 2), padding="same")
                self.disc_l2 = tf.layers.Conv2D(
                    1, [4, 4], strides=(2, 2), padding="same")
                self.disc_l3 = tf.layers.Dense(1)

            self.disc_layers = [self.disc_l1, self.disc_l2, self.disc_l3]

            # gen->disc

            with tf.name_scope("gen-disc"):
                self.gdisc = self.disc(tf.reshape(self.ngen, [-1, 28, 28, 1]))

            # real->disc
            with tf.name_scope("real-disc"):
                self.real = tf.placeholder(
                    dtype="float32", shape=[None, 28, 28], name="real")
                self.rdisc = self.disc(tf.reshape(self.real, [-1, 28, 28, 1]))

            # Generator loss
            with tf.name_scope("gen-loss"):
                self.gloss = tf.losses.sigmoid_cross_entropy(
                    tf.zeros_like(self.gdisc), self.gdisc)
                tf.summary.scalar("Generator loss", self.gloss)

            # Disc loss
            with tf.name_scope("disc-loss"):
                self.dloss_g = tf.losses.sigmoid_cross_entropy(
                    tf.ones_like(self.gdisc), self.gdisc)
                self.dloss_r = tf.losses.sigmoid_cross_entropy(
                    tf.zeros_like(self.rdisc), self.rdisc)
                self.dloss = self.dloss_g + self.dloss_r

                tf.summary.scalar("Discrimnator loss", self.dloss)
                tf.summary.scalar("Discrimnator loss (gen)", self.dloss_g)
                tf.summary.scalar("Discrimnator loss (real)", self.dloss_r)

            # Accuracy
            with tf.name_scope("gen-acc"):
                self.gacc = tf.reduce_mean(1 - self.gdisc)
                tf.summary.scalar("Generator acc", self.gacc)

            with tf.name_scope("disc-acc"):
                self.dacc_g = tf.reduce_mean(self.gdisc)
                self.dacc_r = tf.reduce_mean(1 - self.rdisc)
                self.dacc = (self.dacc_g + self.dacc_r) / 2

                tf.summary.scalar("Discrimnator acc", self.dacc)
                tf.summary.scalar("Discrimnator acc (gen)", self.dacc_g)
                tf.summary.scalar("Discrimnator acc (real)", self.dacc_r)

            # Generator optimizer
            with tf.name_scope("gen-opt"):
                self.goptimizer = tf.train.AdamOptimizer(learning_rate=self.lr_gen)
                self.train_g = self.goptimizer.minimize(
                    self.gloss, var_list=[l.weights for l in self.gen_layers])

            # Disc optimizer
            with tf.name_scope("disc-opt"):
                self.doptimizer = tf.train.AdamOptimizer(learning_rate=self.lr_disc)
                self.train_d = self.doptimizer.minimize(
                    self.dloss, var_list=[l.weights for l in self.disc_layers])

            self.saver = tf.train.Saver()
            self.merged = tf.summary.merge_all()

        self.fw.add_graph(self.graph)

    def disc(self, inp):
        with self.graph.as_default():
            dout = self.disc_l1(inp)
            dout = self.disc_l2(dout)
            dout = tf.nn.relu(dout)

            dout = tf.reshape(dout, [-1, 7 * 7])
            dout = self.disc_l3(dout)

            dout = tf.reshape(dout, [-1])
            dout = tf.nn.sigmoid(dout)

            return dout

    def gen(self, noise):
        with self.graph.as_default():
            gout = self.gen_l1(noise)

            gout = tf.nn.relu(gout)
            gout = tf.reshape(gout, [-1, 7, 7, 1])

            gout = self.gen_l2(gout)
            gout = self.gen_l3(gout)

            gout = tf.nn.relu(gout)
            gout = tf.reshape(gout, [-1, 28, 28])

            return gout

    def load(self):
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())

            try:
                print("Loading...")
                self.saver.restore(self.sess, self.savepath)
                print("Found!")
            except ValueError as e:
                print("Not found!")
                pass

    def train(self, epochs=50):
        with self.graph.as_default():
            test_noise = np.random.normal(0, 1, size=[BATCH_SIZE, NOISE_SIZE])
            for i in range(epochs):
                random.shuffle(TEST_IMAGES)

                gloss_val, dloss_val, dloss_g_val, dloss_r_val, merged_val = self.sess.run(
                    (self.gloss, self.dloss, self.dloss_g, self.dloss_r,
                     self.merged), {
                         self.real: TEST_IMAGES[:BATCH_SIZE],
                         self.gnoise: test_noise
                     })

                gacc_val, dacc_val, dacc_g_val, dacc_r_val = self.sess.run(
                    (self.gacc, self.dacc, self.dacc_g, self.dacc_r), {
                        self.real: TEST_IMAGES[:BATCH_SIZE],
                        self.gnoise: test_noise
                    })

                print("Gen loss: {:.5}, disc loss: {:.5} (g: {:.3}, r: {:.3})".
                      format(gloss_val, dloss_val, dloss_g_val, dloss_r_val))

                print("Gen acc: {:.5}, disc acc: {:.5} (g: {:.3}, r: {:.3})".
                      format(gacc_val, dacc_val, dacc_g_val, dacc_r_val))

                print("Epoch", self.sess.run(self.epoch))

                self.fw.add_summary(merged_val, self.sess.run(self.epoch))

                for batch_idx in range(len(TRAIN_IMAGES) // BATCH_SIZE):
                    batch_x = TRAIN_IMAGES[batch_idx * BATCH_SIZE:
                                           (batch_idx + 1) * BATCH_SIZE]
                    noise = np.random.normal(
                        0, 1, size=[BATCH_SIZE, NOISE_SIZE])

                    _ = self.sess.run(self.train_d, {
                        self.gnoise: noise,
                        self.real: batch_x
                    })

                    _ = self.sess.run(self.train_g, {self.gnoise: noise})

                self.sess.run(self.epoch.assign_add(1))

                self.saver.save(self.sess, self.savepath)


if __name__ == "__main__":
    saves = sorted(os.listdir(SAVE_PATH))

    if len(saves) == 0:
        for lr_gen in [0.0015, 0.002, 0.0025, 0.003]:
            for lr_disc in [0.0015, 0.002, 0.0025]:
                for i in range(3):
                    name = "gen=%.6f_disc=%.6f_%d" % (lr_gen, lr_disc, i)
                    gan = GAN(name, lr_gen, lr_disc)
                    gan.load()
                    gan.train(epochs=20)
        saves = sorted(os.listdir(SAVE_PATH))


    max_epoch = 0
    for f in saves:
        gan = GAN(f)
        gan.load()
        max_epoch = max(gan.sess.run(gan.epoch), max_epoch)

    train_to = max_epoch

    while True:

        train_to += 20

        print("=== Training to %d ==" % train_to)

        for f in saves:
            gan = GAN(f)
            gan.load()
            epoch = gan.sess.run(gan.epoch)
            to_train = train_to - epoch
            print("Training {}, {} epochs".format(f, to_train))
            gan.train(to_train)
