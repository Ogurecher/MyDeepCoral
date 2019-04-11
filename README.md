# Инструкция
## settings.py
Содержит настройки и параметры сети

batch_size - сколько картинок кормим сетке за раз. Чем больше, тем лучше, но при больших значениях лагает. Также при обучении отбрасывается последний batch, т.е. в идеале количество тренировочных данных должно делиться на batch_size с маленьким остатком или без него.

epoch - количество эпох. Чем больше, тем дольше обучается модель.

lr - learning rate. Его я, вроде, более-менее нормально подобрал, но если очень сильно скачет ошибка после нескольких эпох, можно немного снизить.

use_checkpoint - использует данные модели из checkpoint.tar. При включении можно дообучить обученную модель.

## Как запускать?
Для запуска ResNet - ResNet_main.py

Для запуска DeepCORAL - DeepCoral.py

Сначала должно вывести данные о модели, потом должны пойти эпохи с процентами и loss'ом. После каждой эпохи выводятся результаты тестов сначала на тренировочных данных, потом на тестовых.

Когда отработают все эпохи (settings.epoch), должно вывести график. Его скриним и отправляем в беседу. После этого ЗАКРЫВАЕМ ГРАФИК. Только после того, как он закроется, обученная модель сохранится.

Далее создаем папку, называем ее осмысленным названием (сеть, число эпох, batch_size) и копируем туда файлы training_statistic.pkl, testing_statistic.pkl и checkpoint.tar. Не забудьте, что при следующем запуске автоматически будет использоваться этот новый чекпоинт, так что если вы этого не хотите, замените его на нужный в корневой папке.

## Как работать с данными?
Чтобы добавить картинки, вам необходимо: папка images с картинками и файл labels.csv с разметкой.

В папке dataset/labels_and_script лежат: скрипт split.py, который нужен для разделения картинок из images на две папки, разметки наших тренировочных (labels.csv) и тестовых (ResultTest (1).csv) данных.

Чтобы добавить тренировочные или тестовые данные, заходим в dataset/source или dataset/target соответственно, добавляем туда папку images с вашими картинками, скрипт split.py и вашу разметку labels.csv, а затем запускаем из этой папки split.py. После отработки пустую папку images можно удалить.

На данный момент у нас используются картинки 64х64, но сетка работает минимум на 224х224, так что наши картинки доведены до этого размера добавлением черной рамки толщиной 80px.

Чтобы работали картинки 224х224, нужно закомментить в ResNet.py строчку x = self.resize(x)

## Как посмотреть статистику?
Статистика обучения и тестов сохраняется в формате pkl. Чтобы с ним работать, нужно извлечь данные при помощи скрипта unpickle.py. В нем есть методы для конвертации в txt и для построения графика accuracy. Можно добавлять в этот скрипт свои методы по аналогии с уже написанными, чтобы построить любой график или сделать что-то еще со статистикой.
