from model import *
import matplotlib.pyplot as plt

saves = sorted(os.listdir(SAVE_PATH))
for i, f in enumerate(saves):
    print("{}) {}".format(i, f))

choice = saves[int(input("> "))]

g = GAN(choice)
g.load()

noise = np.random.normal(0, 1, size=[15, NOISE_SIZE])
generated, dsc = g.sess.run((g.ngen, g.gdisc), {g.gnoise: noise})

print(dsc.shape)

for i in range(len(generated)):
    plt.subplot(5, 6, (i // 6) * 12 + (i % 6) + 1)
    plt.imshow(generated[i])
    plt.title("{:.2}".format(dsc[i]))

plt.show()

random.shuffle(TEST_IMAGES)

ims = TEST_IMAGES[:15]
dsc = g.sess.run(g.rdisc, {g.real: ims})

for i in range(len(ims)):
    plt.subplot(5, 6, (i // 6) * 12 + (i % 6) + 1)
    plt.imshow(ims[i])
    plt.title("{:.2}".format(dsc[i]))

plt.show()
