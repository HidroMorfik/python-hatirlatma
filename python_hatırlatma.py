# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 22:42:52 2022

@author: HidroMorfik
"""
import github
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/HidroMorfik/python-hatirlatma.git
git push -u origin main

# %% spyder tanıtımı 

print("hello")

# %% degiskenler (veriable)

tamsayi_degisken = 10 
ondalikli_sayi = 12.3

print(tamsayi_degisken)
print(ondalikli_sayi)

# 4 islem özellikleri

pi_sayisi = 3.14
katsayi = 2 

toplam = pi_sayisi + 1
fark = pi_sayisi - 1 
carpma = pi_sayisi * katsayi
bolme = pi_sayisi * katsayi

#print
print("Toplam: ", toplam)
print("Toplam: {} ve fark: {}".format(toplam, fark))
print("Carpma: %.1f, bölme: %.4f"% (carpma, bolme))

# değişkenler arası dönüşüm

carpma_int = int(carpma)
print(carpma_int)

tamsayi_float = float(tamsayi_degisken)
print(tamsayi_float)

# string: karakter dizileri
string= "merhaba dünya"
print(string)

resim_yolu = "veri" + "\\" + "img" + ".png"
print(resim_yolu)


# %% python temel sözdizimi

temel = 6 
TEMEL  = 7 

# yorum 

"""
bu bölümde sozdizimi
    - büyük küçük harf
    - yorum
    - girinti
    - anahtar kelimeler
"""

# girintiler

if 5 < 10 :
    print("yes")
else:
    print("no")


#anahtar kelimeler

de = 4 
# def = 4 
    
# sayili degisken 

sayi1 = 5
sayi2 = 2

# 1sayi = 7

# %% listeler

"""
- bileşik veri türüdür ve çok yönlüdür
- [1, "a", 1.0]
-farklı veri tiplerini barındırabilir
"""

liste = [1,2,3,4,5,6]
print(type(liste))
    
hafta = ["pazartesi", "salı", "çarşamba", "perşembe", "cuma", "cumartesi", "pazar"] 
#ilk eleman
print(hafta[0])  
    
#son eleman
print(hafta[6])  
print(len(hafta)) ,
print(hafta[len(hafta)- 1])
print(hafta[-1])

# liste 2-3-4: 1,2,3 indeks

print(hafta[1:4]) #1'den 4'e kadar 1 dahil - 4 dahil değil    
    
# sayı listesi

sayi_listesi = [1,2,3,4,5,6]
sayi_listesi.append(7) 
print(sayi_listesi)

# listeden eleman silme

sayi_listesi.remove(4)
print(sayi_listesi)

# listeyi ters çevir
sayi_listesi.reverse()
print(sayi_listesi)   

# listeyi sırala
sayi_listesi = [1,3,2,6,5,7,4]
sayi_listesi.sort()
print(sayi_listesi)    

# %% tuple
"""
-değiştirilemez ve sıralı bir veri tipidir
-(1,2,3)
"""
tuple_veritipi = (1,2,3,3,4,5,6)
#ilk elemanı

print(tuple_veritipi[0]) 

# 2. indeksten sonraki elemanları yazdır

print(tuple_veritipi[2:])

# count eleman
print(tuple_veritipi.count(3))

tuple_xyz = (1,2,3)

x, y, z = tuple_xyz

print(x, y, z)
   
# %% deque

from collections import deque

dq = deque(maxlen = 3)
    
dq.append(1) #sonuna 1 ekle [1]
print(dq)

dq.append(2) #sonuna 2 ekle [1,2]
print(dq)

dq.append(3) #sonuna 3 ekle [1,2,3]
print(dq)

dq.append(4) #sonuna 4 ekle [1,2,4]
print(dq)    
    
dq = deque(maxlen = 3)    
    
dq.append(1) #sonuna 1 ekle [1]
print(dq)    
    
dq.append(2) #sonuna 2 ekle [1,2]
print(dq)

dq.appendleft(3) #başına 3 ekle [3,1,2]
print(dq)  
    

dq.clear()
print(dq)

# %% sözlük (dictionary)

"""
- bir çeşit karma tablo türüdür
- anahtar(key) ve değer(value) çiftlerinden oluşur
- {"anahtar": değer}
"""

dictionary = {"istanbul": 34,
              "izmir"   : 35,
              "konya"   : 42}

print(dictionary)

# istanbul anahtarının değeri
print(dictionary["istanbul"])

#anahtarlar keys
print(dictionary.keys())

#değerler values
print(dictionary.values())

# %% koşullu ifadeler if else statement

"""
- bir bool ifadesine göre doğru ya da yanlış olarak değerlendirmesine
  bağlı olarak farklı hesaplamalar veya eylemler gerçekleştiren 
  bir ifadedir.

"""
# büyük küçük sayı karşılaştırması 
sayi1 = 20.0
sayi2 = 20.0
if sayi1 < sayi2:
    print("sayi1 küçüktür sayi2")
    
elif sayi1 > sayi2:
    print("sayi1 büyüktür sayi2")
else:
    print("sayi1 eşittir sayi2")
    

# listelerde if-else
liste = [1,2,3,4,5]
deger = 32
if deger in liste:
    print("{} değeri listenin içerisindedir".format(deger))
else:
    print("{} değeri listenin içerisinde değildir".format(deger))


# sözlüklerde if-else
dictionary = {"Türkiye"   : "Ankara",
              "İngiltere" : "Londra",
              "İspanya"   : "Madrid"}
keys = dictionary.keys()
deger = "Türkiye"
if deger in keys:
    print("evet")
else: print("hayır")


# booleanlarda if-else
bool1 = True
bool2 = False
if bool1 and  bool2: print("Doğru")
else: print("Yanlış")


# %% döngüler (loops)

"""
- Bir dizi üzerine yineleme yapmak için kullanılan yapılardır.
- diziler: liste, tuple, string, sözlük, numpy, pandas veri tipleri
"""

# for 
for i in [1,2,3,4]:
    print(i)

for i in range(1,11):
    print(i)
    
liste = [1,4,5,6,8,3,3,4,67]

print(sum(liste))

toplam = 0
for c in liste:
    toplam = toplam + c
print(toplam)


# while
i = 0 
while i < 4 :
    print(i)
    i = i + 1


liste = [1,4,5,6,8,3,3,4,67]
sinir = len(liste)

her = 0 
hesapla = 0
while her < sinir:
     hesapla = hesapla + liste[her]
     her = her + 1
print(hesapla)


# %% fonksiyonlar

"""
- karmaşık işlemleri toplar ve tek adımda yapmamızı sağlar
- şablon
"""

# user defined function (kullanıcı tarafından tanımlanan fonksiyonlar)

def daireAlan(r):
    """
    Parameters
    ----------
    r : int - daire yarıçapı

    Returns
    -------
    daire_alani : float - daire alanı
    """
    pi = 3.14
   
    daire_alani = pi*(r**2)
    #print(daire_alani)
    
    return daire_alani

dairealanıDegiskeni = daireAlan(3)

print(dairealanıDegiskeni)

def daireCevre(r, pi=3.14):
    """
    
    Parameters
    ----------
    r : int - daire yarıçapı
    pi : float - pi sayısı (3.14).

    Returns
    -------
    daire_cevre : float - daire çevresi
        

    """
    
    daire_cevre = 2*pi*r
    return daire_cevre

daire_cevresi = daireCevre(3)
print(daire_cevresi)

katsayi = 5
def katsayiCarpimi():
    global katsayi
    print(katsayi*katsayi)
katsayiCarpimi()
print(katsayi)

# boş fonksiyon

def bos():
    pass


# built-in function (hazır işlevler)
liste = [1,2,3,4]
print(len(liste))
print(str(liste))
liste2 = liste.copy()
print(liste2)
print(max(liste2))
print(min(liste2))


# lambda functions
"""
- ileri seviyeli 
- küçük ve anonim bir işlemdir
"""

def carpma(x,y,z):
    return x*y*z

sonuc = carpma(2,3,4)
print(sonuc)

# aynı işlem with lambda

func_lambda = lambda x,y,z : x*y*z
sonuc2 = func_lambda(2,3,4)
print(sonuc2)


# %% yield

"""
- iterasyon = yineleme
- generator
- yield
"""
liste = [1,2,3,4]

for i in liste:
    print(i)

"""
generator yineleyecileri
generator değerleri bllekte saklamaz yeri gelince anında üretirler
"""
generator = (x for x in range(1,4))
for i in generator:
    print(i)
    
    
"""
fonksiyon eğer return olarak generator döndürecek ise bunu return yerine yield anahtar keyi ile yapar
"""

def createGenerator():
    liste = range(1,4)
    for i in liste:
        yield i 
generator = createGenerator()
print(generator)

for i in generator:
    print(i)


# %% numpy library
    
"""
- matrisler için hesaplama kolaylığı sağlar
"""
import numpy as np

# 1*15 boutunda bir array(dizi)

dizi = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
print(dizi)

print(dizi.shape) # arrya'in boyutu

dizi2 = dizi.reshape(3,5)
print(dizi2)

print("Şekil: ", dizi2.shape)
print("Boyut: ", dizi2.ndim)
print("veri Tipi: ", dizi2.dtype.name)
print("Boy: ", dizi2.size)

# array type
dizi2D = np.array([[1,2,3,4],[5,6,7,8],[9,8,7,5]])
print(dizi2D)

# sıfırlardan oluşan bir array
sifir_dizi = np.zeros((3,4))
print(sifir_dizi)

# birlerden oluşan bir array
bir_dizi = np.ones((3,4))
print(bir_dizi)

# bos array (sıfıra çok yakın değerler)
bos_dizi = np.empty((3,4))
print(bos_dizi)

# arange(x,y,basamak)
dizi_aralik = np.arange(10,50,5)
print(dizi_aralik)

# linspace(x,y,basamak)
dizi_bosluk = np.linspace(10,20,5)
print(dizi_bosluk)

# float array
float_array = np.float32([[1,2],[3,4]])
print(float_array)

# matematiksel işlemler
a = np.array([1,2,3])
b = np.array([4,5,6])

print(a+b)
print(a-b)
print(a**2)

# dizi eleman toplama 
print(np.sum(a))

# max
print(np.max(a))

# min
print(np.min(a))

# mean ortalama 
print(np.mean(a))

# median ortalama (ortadakş değer)
print(np.median(a))

# rastgele sayi üretme [0,1] arasında sürekli uniform 3*3
random_dizi = np.random.random((3,3))
print(random_dizi)

# indeks
dizi = np.array([1,2,3,4,5,6,7])
print(dizi[0:4])

# dizinin tersi 
print(dizi[::-1])

#
dizi2D = np.array([[1,2,3,4,5],[6,7,8,9,10]])
print(dizi2D)

# dizinin 1. satır ve 1.sütununda bulunan elamnı
print(dizi2D[1,1])

#1.sütun ve tüm satırlar
print(dizi2D[:,1])

# satır 1 , sütun 1,2,3
print(dizi2D[1,1:4])

# dizinin son satır tüm sütunları
print(dizi2D[-1, :])

#
dizi2D = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(dizi2D)

# vektör haline getirme 
vektor = dizi2D.ravel()
print(vektor)

maksimum_sayisinin_indeksi = vektor.argmax()
print(maksimum_sayisinin_indeksi)


 # %% pandas library

"""
- hızlı güçlü ve esnek
"""

import pandas as pd 

# sözlük oluştur
dictionary = {"isim" :["ali","veli","mahmut","ayse","hilal","murat"],
              "yas"  :[15,16,17,33,45,66],
              "maas" :[100,150,240,350,110,220]}


veri = pd.DataFrame(dictionary)
print(veri)


#ilk 5 satır
print(veri.head())

# veri sütunları
print(veri.columns)

# veri bilgisi 
print(veri.info())

# istatistiksel özellikler
print(veri.describe())

#yas sütunu
print(veri["yas"])

#sürun eklemek
veri["şehir"] = ["Akara", "İstanbul", "Konya", "İzmir", "Bursa","Antalya"]
print(veri) 

# yas sütunu
print(veri.loc[:,"yas"])

# yas ve sehir arası sütunu ve 3 satır
print(veri.loc[:2,"yas":"şehir"])

# yas ve sehir arası sütunu ve 3 satır
print(veri.loc[:2,["yas","isim"]])

#satırları tersten yazdır
print(veri.loc[::-1,:])

#yas sütunu with iloc
print(veri.iloc[:,1])

#ilk 3 satır ve yas ve isim
print(veri.iloc[:3,[0,1]])



# filtreleme
dictionary = {"isim" :["ali","veli","mahmut","ayse","hilal","murat"],
              "yas"  :[15,16,17,33,45,66],
              "sehir" :["Ankara", "İstanbul", "Ankara", "İzmir", "Ankara","Antalya"]}

print(veri)


# yasa göre filtre yas < 22 
filtre1 = veri.yas > 22
filtrelenmiş_veri = veri[filtre1]
print(filtrelenmiş_veri)

# ortalama yas 
ortalama_yas = veri.yas.mean()
veri["YAS_GRUBU"] = ["kucuk" if ortalama_yas > i else "buyuk" for i in veri.yas]
print(veri)


# birlestirme
sozluk1 = {"isim" : ["ali","veli","kenan"],
           "yas"  : [15,16,17],
           "sehir": ["İzmir","Ankara","Konya"]}
veri1 = pd.DataFrame(sozluk1)


sozluk2 = {"isim" : ["murat","ayse","hilal"],
           "yas"  : [33,45,66],
           "sehir": ["Ankara","Ankara","Antalya"]}
veri2 = pd.DataFrame(sozluk2)

#dikey
veri_dikey = pd.concat([veri1,veri2], axis=0)

#yatay
veri_yatay = pd.concat([veri1,veri2], axis=1)


# %% matplotlib 
"""
-görselleştirme
"""
import matplotlib.pyplot as plt
import numpy as np

x = np.array([1,2,3,4])
y= np.array([4,3,2,1])

plt.figure()
plt.plot(x,y, color="red", alpha=0.7, label = "line")
plt.scatter(x,y, color="blue", alpha=0.4, label = "scatter")
plt.title("Matplotlib")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.xticks([0,1,2,3,4,5])
plt.legend()
plt.show()

#
fig, axes = plt.subplots(2,1, figsize=(9,7))
fig.subplots_adjust(hspace = 0.5)

x = [1,2,3,4,5,6,7,8,9,10]
y = [10,9,8,7,6,5,4,3,2,1]

axes[0].scatter(x,y)
axes[0].set_title("sub-1")
axes[0].set_ylabel("sub-1 y")
axes[0].set_xlabel("sub-1 x")

axes[1].scatter(y,x)
axes[1].set_title("sub-2")
axes[1].set_ylabel("sub-2 y")
axes[1].set_xlabel("sub-2 x")

# random resim
plt.figure()
img = np.random.random((50,50))
plt.imshow(img, cmap= "gray") #0(siyah) - 1(beyaz) -> 0.5(gri)
plt.axis("off")
plt.show()

# %% OS (operating system)

import os
print(os.name) # nt -> windows , posix -> linux

currentDir = os.getcwd()
print(currentDir)

# new folder
folder_name = "new_folder"
os.mkdir(folder_name)

new_folder_name = "new_folder2"
os.rename(folder_name, new_folder_name)

os.chdir(currentDir+"\\"+new_folder_name)
print(os.getcwd())

files = os.listdir()

for f in files:
    if f.endswith(".py"):
        print(f)


os.rmdir(new_folder_name)

for i in os.walk(currentDir):
    print(i)
    
os.path.exists("python_hatırlatma.py")
