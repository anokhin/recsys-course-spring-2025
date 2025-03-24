# Botify

Сервис рекомендаций реализован как приложение на [Flask](https://flask-restful.readthedocs.io/en/latest/).
Это приложение умеет обрабатывать запросы по REST API.
В качестве in-memory кэша используется [Redis](https://redis.io/).
В качестве load balancer используется [nginx](https://www.nginx.com/)
Приложение пишет лог событий в json в папку `/app/log/`

![Архитектура сервиса botify](architecture.png)

## Инструкция

1. [Устанавливаем docker](https://www.docker.com/products/docker-desktop)
1. Собираем образы и запускаем контейнеры (2 рекомендера в параллель)
   ```
   docker-compose up -d --build --force-recreate --scale recommender=N
   ```   
1. Смотрим логи рекомендера
   ```
   docker logs botify-recommender-n
   ```
1. Останавливаем контейнеры
   ```
   docker-compose stop
   ```
1. Модифицируем код в этом модуле
1. Повторяем шаги 2-4, пока не достигнем поставленной цели 

## Полезные команды
Проверяем, что сервис жив
```
curl http://localhost:5001/
```
Запрашиваем информацию по треку
```
curl http://localhost:5001/track/42
```
Запрашиваем следующий трек
```
curl -H "Content-Type: application/json" -X POST -d '{"track":10,"time":0.3}'  http://localhost:5001/next/1
```
Завершаем пользовательскую сессию
```
curl -H "Content-Type: application/json" -X POST -d '{"track":10,"time":0.3}'  http://localhost:5001/last/1
```
Скачиваем логи пользовательских сессии с контейнера
```
docker cp botify-recommender-n:/app/log/ /tmp/
```
Скачиваем логи пользовательских сессий со всех контейнеров
```
python dataclient.py --recommender N log2local ~/Desktop/data
```
Мониторим загрузку хостов
```
docker stats botify-recommender-1 botify-nginx-1 redis-container
```

## Работа на удаленном сервере
Заходим на сервер по ssh, прокидываем порт
```
ssh -L 16006:127.0.0.1:30007 dnikanorova@mipt-client.atp-fivt.org
```
Посмотреть, какие порты заняты
```
ss -tulpn
```
Создаем директорию для jupyter ноутбуков
```
mkdir jupyter
```
Отправляем файлы с локальной машины на удаленный сервер
```
scp jupyter/Week1Seminar.ipynb dnikanorova@mipt-client.atp-fivt.org:jupyter
```
Скачиваем файлы с удаленного сервера на локальную машину:
```
scp dnikanorova@mipt-client.atp-fivt.org:data/top_tracks.json ../botify/data/top_tracks.json
```
Стартуем ноутбук
```
export PYSPARK_DRIVER_PYTHON=jupyter
export PYSPARK_PYTHON=/usr/bin/python3.6
export PYSPARK_DRIVER_PYTHON_OPTS='notebook --ip="*" --port=30007 --no-browser'
pyspark2 --master=yarn --num-executors=2
```
Открываем ноутбук на локальном хосте
```
http://localhost:16006/
```

## Создание окружения на удаленном сервере
Работаем с python3.4
```
export PYSPARK_PYTHON=/usr/bin/python3.4
```
Создаем окружение на сервере (используйте именно virtualenv)
```
virtualenv -p /usr/bin/python3.6 envs/mobod-2023
```
Активируем окружение
```
source envs/atp-mobod_2022/bin/activate
```
Устанавливаем зависимости
```
cp /home/dnikanorova/envs/requirements.txt <your_dir>
pip install -r requirements.txt
```
Добавляем созданный кернел для jupyter ноутбука 
```
python -m ipykernel install --user --name=atp-mobod_2022
```
После этого запускаем jupyter notebook как обычно, выбираем новый kernel:
Kernel > Change kernel > atp-mobod_2022

## Запускаем Tensor Board
Заходим на кластер, прокидывая порт 6006 (tensorboard по умолчанию открывается именно на нем)
```
ssh -L 16007:127.0.0.1:6006  dnikanorova@mipt-client.atp-fivt.org
```
Запускаем tensorboard, указывая директорию с логами в параметре --logdir
```
tensorboard --logdir data/tb
```
## Работа на кластере HDFS
![Архитектура сервиса botify](hdfs.png)
Закидываем логи контейнера в hdfs (папка script должна быть в $PYTHONPATH). 
Программа dataclient.py отправляет данные в папку юзера: /user/dnikanorova/
Аргумент --recommender задает кол-во контейнеров, с которых вы хотите собрать логи. 
Выставляйте его равным запущенному кол-ву докер-контейнерв botify_recommender
```
python dataclient.py --user dnikanorova --recommender N log2hdfs --cleanup my_remote_dir
```
Отправляем файлы с сервера в hdfs
```
hadoop fs -put my_remote_dir/random_10k.json /user/dnikanorova/my_remote_dir/
```
Проверяем содержимое папки в hdfs
```
hadoop fs -ls /user/dnikanorova/my_remote_dir
```
Смотрим размер файлов в директории
```
hadoop fs -du -h /user/dnikanorova/my_remote_dir
```