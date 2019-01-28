# Fix ssl
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import random
import tensorflow as tf
import numpy as np
import os

SAVE_PATH = os.join(["saves", "networks"])
LOG_PATH = os.join(["saves", "log"])


def trait_network(idnr, epochs, lr_gen, lr_disc):
    g = tf.Graph()

    with g.as_default():
        savepath = os.path.join(SAVE_PATH, str(idnr))
        logpath = os.path.join(LOG_PATH, str(idnr))

        fw = tf.summary.FileWriter(logpath)

        epoch = tf.Variable(0, name="epoch")

        BATCH_SIZE = 300
        NOISE_SIZE = 30

        dataset = tf.keras.datasets.mnist
        (train_images, _), (test_images, _) = dataset.load_data()

        # Generator def

        with tf.name_scope("generator"):
            gen_l1 = tf.layers.Dense(7 * 7)
            gen_l2 = tf.layers.Conv2DTranspose(
                8, [4, 4], strides=(2, 2), padding="same")
            gen_l3 = tf.layers.Conv2DTranspose(
                1, [6, 6], strides=(2, 2), padding="same")

        gen_layers = [gen_l1, gen_l2, gen_l3]


        def gen(noise):
            gout = gen_l1(noise)

            gout = tf.nn.relu(gout)
            gout = tf.reshape(gout, [-1, 7, 7, 1])

            gout = gen_l2(gout)
            gout = gen_l3(gout)

            gout = tf.nn.relu(gout)
            gout = tf.reshape(gout, [-1, 28, 28])

            return gout


        # noise->gen

        with tf.name_scope("noise-gen"):
            gnoise = tf.placeholder(
                dtype="float32", shape=[None, NOISE_SIZE], name="noise")
            ngen = gen(gnoise)

            tf.summary.image("generated", tf.reshape(ngen, [-1, 28, 28, 1]))

        # Discriminator def

        with tf.name_scope("discriminator"):
            disc_l1 = tf.layers.Conv2D(8, [4, 4], strides=(2, 2), padding="same")
            disc_l2 = tf.layers.Conv2D(1, [4, 4], strides=(2, 2), padding="same")
            disc_l3 = tf.layers.Dense(1)

        disc_layers = [disc_l1, disc_l2, disc_l3]


        def disc(inp):
            global disc_l1, disc_l2, disc_l3

            dout = disc_l1(inp)
            dout = disc_l2(dout)
            dout = tf.nn.relu(dout)

            dout = tf.reshape(dout, [-1, 7 * 7])
            dout = disc_l3(dout)

            dout = tf.reshape(dout, [-1])
            dout = tf.nn.sigmoid(dout)

            return dout


        # gen->disc

        with tf.name_scope("gen-disc"):
            gdisc = disc(tf.reshape(ngen, [-1, 28, 28, 1]))

        # real->disc
        with tf.name_scope("real-disc"):
            real = tf.placeholder(dtype="float32", shape=[None, 28, 28], name="real")
            rdisc = disc(tf.reshape(real, [-1, 28, 28, 1]))

        # Generator loss
        with tf.name_scope("gen-loss"):
            gloss = tf.losses.sigmoid_cross_entropy(tf.zeros_like(gdisc), gdisc)
            tf.summary.scalar("Generator loss", gloss)

        # Disc loss
        with tf.name_scope("disc-loss"):
            dloss_g = tf.losses.sigmoid_cross_entropy(tf.ones_like(gdisc), gdisc)
            dloss_r = tf.losses.sigmoid_cross_entropy(tf.zeros_like(rdisc), rdisc)
            dloss = dloss_g + dloss_r

            tf.summary.scalar("Discrimnator loss", dloss)
            tf.summary.scalar("Discrimnator loss (gen)", dloss_g)
            tf.summary.scalar("Discrimnator loss (real)", dloss_r)

        # Accuracy
        with tf.name_scope("gen-acc"):
            gacc = tf.reduce_mean(1 - gdisc)
            tf.summary.scalar("Generator acc", gacc)

        with tf.name_scope("disc-acc"):
            dacc_g = tf.reduce_mean(gdisc)
            dacc_r = tf.reduce_mean(1 - rdisc)
            dacc = (dacc_g + dacc_r) / 2

            tf.summary.scalar("Discrimnator acc", dacc)
            tf.summary.scalar("Discrimnator acc (gen)", dacc_g)
            tf.summary.scalar("Discrimnator acc (real)", dacc_r)

        # Generator optimizer
        with tf.name_scope("gen-opt"):
            goptimizer = tf.train.AdamOptimizer(learning_rate=lr_gen)
            train_g = goptimizer.minimize(
                gloss, var_list=[l.weights for l in gen_layers])

        # Disc optimizer
        with tf.name_scope("disc-opt"):
            doptimizer = tf.train.AdamOptimizer(learning_rate=lr_disc)
            train_d = doptimizer.minimize(
                dloss, var_list=[l.weights for l in disc_layers])

        saver = tf.train.Saver()
        merged = tf.summary.merge_all()


        def load(sess):
            sess.run(tf.global_variables_initializer())

            try:
                print("Loading...")
                saver.restore(sess, SAVE_PATH)
                print("Found!")
            except ValueError as e:
                print("Not found!")
                pass

        if __name__ == "__main__":
            with tf.Session() as sess:
                load(sess)

                fw.add_graph(sess.graph)

                test_noise = np.random.normal(0, 1, size=[BATCH_SIZE, NOISE_SIZE])
                for i in range(EPOCHS):
                    random.shuffle(test_images)

                    gloss_val, dloss_val, dloss_g_val, dloss_r_val, merged_val = sess.run(
                        (gloss, dloss, dloss_g, dloss_r, merged), {
                            real: test_images[:BATCH_SIZE],
                            gnoise: test_noise
                        })

                    gacc_val, dacc_val, dacc_g_val, dacc_r_val = sess.run(
                        (gacc, dacc, dacc_g, dacc_r), {
                            real: test_images[:BATCH_SIZE],
                            gnoise: test_noise
                        })

                    print("Gen loss: {:.5}, disc loss: {:.5} (g: {:.3}, r: {:.3})".
                          format(gloss_val, dloss_val, dloss_g_val, dloss_r_val))

                    print("Gen acc: {:.5}, disc acc: {:.5} (g: {:.3}, r: {:.3})".
                          format(gacc_val, dacc_val, dacc_g_val, dacc_r_val))

                    print("Epoch", sess.run(epoch))

                    fw.add_summary(merged_val, sess.run(epoch))

                    # Train disc
                    for batch_idx in range(len(train_images) // BATCH_SIZE):
                        batch_x = train_images[batch_idx * BATCH_SIZE:(batch_idx + 1) *
                                               BATCH_SIZE]
                        noise = np.random.normal(0, 1, size=[BATCH_SIZE, NOISE_SIZE])

                        _ = sess.run(train_d, {gnoise: noise, real: batch_x})

                        _ = sess.run(train_g, {gnoise: noise})

                    sess.run(epoch.assign_add(1))

                    saver.save(sess, SAVE_PATH)
