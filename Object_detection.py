import tensorflow as tf
import numpy as np
import cv2
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from tensorflow import keras
from keras.layers import Dense, Flatten, Input, Conv2D, Conv2DTranspose, Concatenate, LeakyReLU, Dropout

print('парсим данные ', '_-*50')
tree = ET.parse('D:\\images_2\\test_0.xml')  # адрес файла
root = tree.getroot()  # парсим

print(root[1].tag, root[1].text)
print(root[2].tag, root[2].text)
print(root[4][0].tag, root[4][0].text)
print(root[4][1].tag, root[4][1].text)

# колличество размеченых обьектов
num_objects = len(root) - 6
print(num_objects)

print('самостоятельная загрузка картинки и обработка ее', '_-*50')
# самостоятельная обработка и загрузка картинки
def load_img(img):
    img = tf.io.read_file(img)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32) / 256
    img = tf.image.resize(img, (128, 128))
    return img


tree = ET.parse('D:\\images_2\\test_0.xml')  # адрес файла
root = tree.getroot()  # парсим
num_objects = len(root) - 6
cords = []
w = int(root[4][0].text)  # ширина x
h = int(root[4][1].text)  # высота y

for num in range(num_objects):
    print(root[num + 6][0].text)  # имя обьекта

    object_cords = []
    # нормализуем координаты от -1 до 1, опираясь на исходные координаты
    object_cords.append(int(root[num + 6][4][0].text) / w * 2 - 1)
    object_cords.append(int(root[num + 6][4][1].text) / h * 2 - 1)
    object_cords.append(int(root[num + 6][4][2].text) / w * 2 - 1)
    object_cords.append(int(root[num + 6][4][3].text) / h * 2 - 1)

    cords.append(object_cords)
print(cords)

img = load_img(root[2].text)
plt.figure(figsize=(10, 6))
ax = plt.subplot(3, 1, 1)
plt.imshow(img)
plt.axis('off')
plt.show()

print('преобразуем папку в tf рекорд', '_-*50')# для быстроты работы
# преобразуем в tf рекорд
fn = "D:\\images_2\\"
# формируем список всех xml файлов в папке
p = [fn + '/' + f for f in listdir(fn) if isfile(join(fn, f)) and f[-1] == 'l']
print(p[:10])


def load_img(img): # пишем в формате тензорфло для быстроты рабооты
    img = tf.io.read_file(img)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32) / 256
    img = tf.image.resize(img, (128, 128))
    return img



writer = tf.io.TFRecordWriter('bounding_box_dataset.tfrecord') # просто сохдаем файл, куда будем записывать tf рекордер

for xml in p:
    tree = ET.parse(xml)  # адрес файла
    root = tree.getroot()  # парсим
    num_objects = len(root) - 6
    cords = []
    w = int(root[4][0].text)  # ширина x
    h = int(root[4][1].text)  # высота y
    for num in range(num_objects):
        object_cords = []
        # нормализуем координаты от -1 до 1, опираясь на исходные координаты
        object_cords.append(int(root[num + 6][4][0].text) / w * 2 - 1)
        object_cords.append(int(root[num + 6][4][1].text) / h * 2 - 1)
        object_cords.append(int(root[num + 6][4][2].text) / w * 2 - 1)
        object_cords.append(int(root[num + 6][4][3].text) / h * 2 - 1)
        cords.append(object_cords)

    img = load_img(root[2].text)
    # переводим данные в байты, что необходимо для tf рекорд
    serialized_img = tf.io.serialize_tensor(img).numpy()
    serialized_cords = tf.io.serialize_tensor(cords).numpy()
    # собираем экзепмляр (тренеровочный)
    example = tf.train.Example(features=tf.train.Features(feature={
        'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized_img])),
        'cords': tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized_cords]))
    }))


    writer.write(example.SerializeToString()) # записываем записываем экземпляр в запись

writer.close()

dataset = tf.data.TFRecordDataset('bounding_box_dataset.tfrecord') # читаем наш дата сет


def parse_record(record):
    # нужно описать приходящий экземпляр
    # имена элементов как при записи
    feature_description = {
        'img': tf.io.FixedLenFeature([], tf.string),
        'cords': tf.io.FixedLenFeature([], tf.string)
    }
    parsed_record = tf.io.parse_single_example(record, feature_description)
    img = tf.io.parse_tensor(parsed_record['img'], out_type=tf.float32)
    cords = tf.io.parse_tensor(parsed_record['cords'], out_type=tf.float32)
    return img, cords


# пройдемся по записи и распакуем ее
dataset = dataset.map(parse_record)

# выводим бач изображений и бач координат
for i, c in dataset.take(1):
    print(i.shape)
    print(c.shape)



'''gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)'''


dataset = tf.data.TFRecordDataset('bounding_box_dataset.tfrecord')


def parse_record(record):
    # нужно описать приходящий экземпляр
    # имена элементов как при записи
    feature_description = {
        'img': tf.io.FixedLenFeature([], tf.string),
        'cords': tf.io.FixedLenFeature([], tf.string)
    }
    parsed_record = tf.io.parse_single_example(record, feature_description)
    img = tf.io.parse_tensor(parsed_record['img'], out_type=tf.float32)
    cords = tf.io.parse_tensor(parsed_record['cords'], out_type=tf.float32)
    return img, cords


# пройдемся по записи и распакуем ее
dataset = dataset.map(parse_record)

# еще раз проверим
for i, c in dataset.take(1):
    plt.figure(figsize=(10, 6))
    ax = plt.subplot(3, 1, 1)
    i = i.numpy()
    c = c.numpy()
    c = (c + 1) / 2 * 128 # обратно из от -1...1 к 0...64
    c = c.astype(np.int16)  # для opencv
    for bb in c:
        i = cv2.rectangle(i, (bb[0], bb[1]), (bb[2], bb[3]), (1, 0, 0), 1)
    plt.imshow(i)
    plt.show()
    print(c)

dataset = dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE).batch(32).shuffle(50)

inputs = Input((128, 128, 3))
x = Conv2D(32, 3, activation='relu', padding='same')(inputs)
x = Conv2D(64, 3, activation='relu', padding='same', strides=2)(x)
x = Conv2D(64, 3, activation='relu', padding='same')(x)
x = Conv2D(128, 3, activation='relu', padding='same', strides=2)(x)
x = Conv2D(128, 3, activation='relu', padding='same')(x)
x = Conv2D(128, 3, activation='relu', padding='same', strides=2)(x)

x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(128, activation='relu')(x)
x = Dense(4)(x)

outputs = x

boxregressor = keras.Model(inputs, outputs)


# расчет ошибки
def IoU_Loss(true, pred):
    # (32, 1, 4)
    t1 = true
    t2 = pred

    # наши данные уже в правильном порядке
    minx1, miny1, maxx1, maxy1 = tf.split(t1, 4, axis=2)

    # minx1.shape = (32,1,1) работаем сразу с целым батчем

    # а вот нейросеть не знает, где должна быть min и max координата
    # нормализуем данные
    fminx, fminy, fmaxx, fmaxy = tf.split(t2, 4, axis=2)
    minx2 = tf.minimum(fminx, fmaxx)
    miny2 = tf.minimum(fminy, fmaxy)
    maxx2 = tf.maximum(fminx, fmaxx)
    maxy2 = tf.maximum(fminy, fmaxy)

    # считаем пересечение
    x_overlap = tf.maximum(0.0, tf.minimum(maxx1, maxx2) - tf.maximum(minx1, minx2))
    y_overlap = tf.maximum(0.0, tf.minimum(maxy1, maxy2) - tf.maximum(miny1, miny2))

    intersection = x_overlap * y_overlap

    # площади найти не сложно
    area1 = (maxx1 - minx1) * (maxy1 - miny1)
    area2 = (maxx2 - minx2) * (maxy2 - miny2)

    # обьединение тоже по алгоритму
    union = area1 + area2 - intersection

    IoU = intersection / union
    loss = 1.0 - IoU

    return loss


# создаем класс модели
class Model(tf.keras.Model):
    def __init__(self, nn_box):
        super(Model, self).__init__()
        self.nn_box = nn_box

        self.box_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)

    @tf.function
    def training_step(self, x, true_boxes):
        with tf.GradientTape() as tape_box:
            pred = self.nn_box(x, training=True)
            pred = tf.reshape(pred, [-1, 1, 4])  # далее заменим это на [-1, кол-во рамок, 4]

            loss = IoU_Loss(true_boxes, pred)

        grads = tape_box.gradient(loss, self.nn_box.trainable_variables)
        self.box_optimizer.apply_gradients(zip(grads, self.nn_box.trainable_variables))

        return loss


model = Model(boxregressor)
for i, c in dataset.take(1):
    print(tf.reduce_mean(model.training_step(i, c)))


# model.nn_box.load_weights('bounding_box_for_one_object.h5')
# проверка работы
def testing():
    for ii, cc in dataset.take(1):
        # обрабатывем целый батч, используем только пять элементов
        pred = model.nn_box(ii)
        plt.figure(figsize=(16, 10))

        for num in range(10):
            i = ii[num]

            pred = tf.reshape(pred, [-1, 1, 4])
            c = pred[num]

            ax = plt.subplot(1, 10, num + 1)
            # переход в numpy для работы в opencv
            i = i.numpy()
            c = c.numpy()
            c = (c + 1) / 2 * 128  # обратно из от -1...1 к 0...64
            c = c.astype(np.int16)  # для opencv
            for bb in c:
                i = cv2.rectangle(i, (bb[0], bb[1]), (bb[2], bb[3]), (0, 1, 0), 1)
            plt.imshow(i)

        plt.show()
        print(c)


testing()

# обучение
from IPython.display import clear_output

hist = np.array(np.empty([0]))
epochs = 15
for epoch in range(1, epochs + 1):
    loss = 0
    lc = 0
    for step, (i, c) in enumerate(dataset):
        loss += tf.reduce_mean(model.training_step(i, c))
        lc += 1
    clear_output(wait=True)
    print(epoch)
    hist = np.append(hist, loss / lc)
plt.plot(np.arange(0, len(hist)), hist)
plt.show()
testing()
# model.nn_box.save('bounding_box_for_one_object.h5')
# model.nn_box.load_weights('bounding_box_for_one_object.h5')
# tree = ET.parse('C:\\Users\\Артем\\images\\test.xml') #адрес файла
