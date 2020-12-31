"""
Bu bölümde mnist dataseti ile çalışıyoruz.
http://yann.lecun.com/exdb/mnist/
Mnist datasetinde 28x28 piksel boyutunda el yazısı rakamlardan oluşan 70.000 resim bulunmaktadır.

Oluşturduğumuz sinir ağı yapabileceğimiz en basit sinir ağı.
Öncelikle mnist resimlerini alıp düzleştiriyoruz ve vektöre dönüştürüyoruz.
Bu şekilde her resim için 784 uzunluğunda bir vektör elde ediyoruz. (28, 28 piksel. 28x28=784)
Vektöre dönüştürdüğümüz resimleri 10 tane nörona bağlıyoruz. (0'dan 9'a 10 tane rakam olduğu için 10 nöron)
Modelimiz bu nöronlardan hangisi en aktifse yani hangi nöronun değeri en büyükse ona göre tahmin yapıyor.
Bu tahmini loss fonksiyonundan geçirip gerçek değere ne kadar uzak olduğunu hesaplıyoruz.
En son optimizasyon ile weight ve bias değerlerini güncelleyerek öğretme işlemini gerçekleştiriyoruz.
Bu işlemi tekrar tekrar binlerce defa tekrarlıyoruz.
"""
#####################################
# Tensorflow kütüphanesini import ediyoruz.
import tensorflow as tf


#####################################
# Mnist datasetini bu satırlar ile indirip kullanıma uygun hale getiriyoruz.
# Dataset proje içerisinde data/MNIST klasörüne indirilecek.
# Etiketleri one hot olarak alıyoruz. (One hot mesela bu şekilde: 6 = [0,0,0,0,0,0,1,0,0,0])
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/MNIST/", one_hot=True)


#####################################
# Placeholder'lar Tensorflow'da yer tutucularımızdır.
# Yani placeholder ile tanımladığımız değişkenlere daha sonra değerler atayacağız.
# x placeholder'ına resimleri atayacağız, y_true placeholder'ına ise etiketleri atayacağız.
# Bu atama işlemlerini feed dict içerisinde yapacağız.
# x'e boyut olarak None ve 784 veriyoruz. None dediğimiz için kaç tane resim gelirse gelsin kabul edecektir.
# kesin bir sayı belirtmiyoruz.
# 784 ise mnist resimlerinin düzleştirilmiş haliydi.
# y_true için None ve 10 atıyoruz. 10 tane sınıf sayımız yani mnist içerisinde 10 tane rakam olduğu için 10 veriyoruz.
x = tf.placeholder(tf.float32, [None, 784])
y_true = tf.placeholder(tf.float32, [None, 10])


#####################################
# Variable ile eğitilecek değişken tanımlıyoruz.
# Weight ve bias eğitilecek parametrelerdir.
# Eğitim esnasında bu parametreler güncellenecek ve böylelikle modelimiz daha isabetli hale gelecek.
# Variable ile tanımladığımızda bu değişkenlerin eğitilecek değişkenler olduğunu bildiriyoruz.
# Boyut olarak 784'e 10 veriyoruz. x*w yapacağız. Burada x'te w'da matris.
# O yüzden çarpma işlemi yapabilmek için sütun satır sayıları eşit olmalı. 784 vermemizin nedeni bu.
# Bu layer'da 10 nöron olacak. (784, 10)
# x*w yaptıktan sonra bias ile toplayacağız. Boyutuna 10 tane nöron olduğu için 10 veriyoruz.
# tf.zeros ile matristeki tüm elemanlara 0 veriyoruz. Eğitim esnasında bunlar değişecek.
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))


#####################################
# Yukarıda input, weight, bias tanımladık
# x*w+b işlemini gerçekleştiriyoruz.
# tf.matmul matrislerde çarpma işlemi yapıyor.
logits = tf.matmul(x, w) + b
# Softmax sınıflandırma yaparken kullandığımız bir aktivasyon fonksiyonu.
# 10 nörondaki tüm değerler 0-1 arasına sıkıştırılacak ve aynı zamanda bu nöronlardaki değerlerin toplamı 1'e eşit olacak.
y = tf.nn.softmax(logits)


#####################################
# Loss fonksiyonu olarak cross entropy kullanıyoruz.
# Loss fonksiyonu tahmin edilen değerin gerçek değere ne kadar uzak olduğunu hesaplayacak.
# Bu fonksiyon bizden logits ve labels bekliyor.
# Logits x*w+b işleminin sonucu. (Bu fonksiyon kendi içinde softmax uyguladığı için y yerine logits veriyoruz.)
# Labels etiketlerimiz. Bunları y_true diye placeholder olarak tanımlamıştık.
xent = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true)
# Resimleri batch olarak yani 128'er 128'er besleceyeğiz.
# Bu yüzden softmax her resim için bir değer verecek. Yani 128 tane değer verecek.
# reduce_mean ile tüm bu değerlerin ortalamasını alıyoruz.
# Bu sayede 0'a yaklaştırılabilecek tek bir değer elde ediyoruz.
loss = tf.reduce_mean(xent)


#####################################
# Modelimiz ne kadar isabetli tahmin yapabiliyor görmek için doğru cevaplar ile tahminleri karşılaştırıyoruz.
# y tahmin, y_true ise doğru olan değerler.
# tf.equal ile bu iki değer birbirine eşit mi değil mi bakıyoruz.
# Tüm bunları True, False olarak bir listeye atıyoruz.
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))
# Reduce mean ile tüm True False'ların ortalamasını alarak isabet oranını hesaplıyoruz.
# True ve False üzerinde matematiksel işlem yapamadığımız için tf.cast ile bunları 1 ve 0'a dönüştürüyoruz.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#####################################
# Optimizasyon yaparak weight ve bias değerlerini güncelliyoruz.
# 0.5 learning rate'imiz.
optimize = tf.train.GradientDescentOptimizer(0.5).minimize(loss)


#####################################
# tf.Session ile Tensorflow Session'ı açıyoruz
# Bu kısma kadar yazdığımız kodların hiçbiri henüz çalışmıyor, sadece tanımlamalar yaptık.
# sess.run ile Session'a ekleyerek çalıştırmamız lazım.
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Resimleri batch olarak alacağız.
batch_size = 128


#####################################
# Asıl eğitim yaptığımız fonksiyon burada.
# Iterations kadar dönecek bir döngü oluşturuyoruz.
# Bu döngü döndükçe model eğitilecek ve öğretmek istediğimiz şeyi öğretecek.
def training_step (iterations):
    for i in range (iterations):
        # mnist.train.next_batch ile datasetinden batch_size kadar yani 128 tane resim ve etiket alıyoruz.
        # Aldığımız resimleri x_batch'e etiketleri ise y_batch'e atıyoruz.
        # Bu döngü her döndüğünde farklı farklı resimleri alarak grafiğe besliyoruz.
        x_batch, y_batch = mnist.train.next_batch(batch_size)
        # Feed dict ile placeholder'lara ne atanacağını belirliyoruz.
        # x ve y_true için placeholder oluşturmuştuk.
        # x'e resimleri y_true'ya etiketleri atıyoruz.
        feed_dict_train = {x: x_batch, y_true: y_batch}
        # sess.run ile yukarda tanımladığımız kodları çalıştırıyoruz.
        # optimize'yi çalıştırarak optimize yapıyoruz. Placeholder'lara ne atadığımızı öğrenmek için feed dict bekleyecektir.
        sess.run(optimize, feed_dict=feed_dict_train)


#####################################
# Eğitim tamamlandıktan sonra model ne kadar başarılı test etmemiz gerekiyor.
# Test için fonksiyon oluşturuyoruz.
def test_accuracy ():
    # Feed dict'te bu sefer x ve y_true'ya test resim ve etiketlerini atıyoruz.
    feed_dict_test = {x: mnist.test.images, y_true: mnist.test.labels}
    # sess.run ile accuracy'yi çalıştırarak bir değişkene atıyoruz.
    # Böylelikle modelimiz ne kadar isabetli çalışıyor göreceğiz.
    acc = sess.run(accuracy, feed_dict=feed_dict_test)
    # acc'yi yazdırıyoz.
    print('Testing accuracy:', acc)


#####################################
# Tanımladığımız fonksiyonları çağırıyoruz.
# 2000 iterasyon eğitim yapıyoruz.
training_step(2000)
test_accuracy()