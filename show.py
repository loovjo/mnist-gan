from model import *
import matplotlib.pyplot as plt

with tf.Session() as sess:
    load(sess)

    noise = np.random.normal(0, 1, size=[15, NOISE_SIZE])
    generated, dsc = sess.run((ngen, gdisc), {gnoise: noise})

    print(dsc.shape)

    for i in range(len(generated)):
        plt.subplot(5, 6, (i // 6) * 12 + (i % 6) + 1)
        plt.imshow(generated[i])
        plt.title("{:.2}".format(dsc[i]))

    plt.show()

    random.shuffle(test_images)

    ims = test_images[:15]
    dsc = sess.run(rdisc, {real: ims})

    for i in range(len(ims)):
        plt.subplot(5, 6, (i // 6) * 12 + (i % 6) + 1)
        plt.imshow(ims[i])
        plt.title("{:.2}".format(dsc[i]))

    plt.show()
