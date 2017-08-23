import matplotlib.pyplot as plt
import tensorflow as tf

# reading images using FIFOQueue within tensorflow graph
def ImageReader(files, channels=0, shuffle=False):
    file_queue = tf.train.string_input_producer(files, shuffle=shuffle)
    reader = tf.WholeFileReader()
    key, value = reader.read(file_queue)
    image = tf.image.decode_image(value, channels=channels)
    image.set_shape([None, None, 3])
    return image

# writing batch of images within tensorflow graph
def ImageBatchWriter(sess, images, files, dtype=tf.uint8):
    pngs = []
    for i in range(len(files)):
        img = images[i]
        img = tf.image.convert_image_dtype(img, dtype, saturate=True)
        png = tf.image.encode_png(img, compression=9)
        pngs.append(png)
    pngs = sess.run(pngs)
    for i in range(len(files)):
        with open(files[i], 'wb') as f:
            f.write(pngs[i])

file = r'I:\Met-Art\[Met-Art] - 2010-12-23 - Virginia Sun - Presenting Virginia by Rylsky (x114) 5000x3333\MET-ART_ry_415_0009.jpg'
file_list = [file]
image = ImageReader(file_list, channels=3, shuffle=True)

dtype = tf.float32
src = image[1300:1900,2300:2900]
src = tf.image.convert_image_dtype(src, dtype)

shape3 = tf.shape(src)
size0 = shape3[0:2]
#size1 = size0 // 2
size1 = size0 * 2
tf.Print(src, [shape3, size0, size1])

down0 = tf.image.resize_images(src, size1, tf.image.ResizeMethod.BILINEAR)
down1 = tf.image.resize_images(src, size1, tf.image.ResizeMethod.NEAREST_NEIGHBOR)
down2 = tf.image.resize_images(src, size1, tf.image.ResizeMethod.BICUBIC)
down3 = tf.image.resize_images(src, size1, tf.image.ResizeMethod.AREA)


with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    
    image_nodes = [src, down0, down1, down2, down3]
    images = sess.run(image_nodes)
    print(len(images))
    print(images[0].shape)
    
    plt.imshow(images[0])
    plt.show()
    plt.imshow(images[1])
    plt.show()
    plt.imshow(images[2])
    plt.show()
    
    out_files = ['0.0.png', '0.bilinear.png', '0.nearest.png', '0.bicubic.png', '0.area.png']
    ImageBatchWriter(sess, image_nodes, out_files)
    
    coord.request_stop()
    coord.join(threads)

