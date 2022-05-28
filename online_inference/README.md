Online inference
==============================

В данном проекте реализован веб-сервис на FastApi, который выгружает модель по ссылке c облачного хранилища (реализована работа с гугл-диск), позволяет делать запросы для получения результатов предсказания модели.  
Сборка docker образа из исходников:
```bash
cd online_inference
docker build -t filivan/online_inference .
```
Загрузка docker образа из docker hub:
```bash
docker pull filivan/online_inference
```
Запуск сервиса (будет доступен на localhost:8000):
```bash
docker run --rm -p 8000:8000 filivan/online_inference
```
Чтобы задать ссылку для скачивания модели:
```bash
docker run --rm -e PATH_TO_MODEL="https://drive.google.com/file/d/1EmcCrbnl1Q-5YCcpC6ohpSJR0wx23zFe/view?usp=sharing" -p 8000:8000 filivan/online_inference
```
Доступные endpoints:
* `/health` - показывает, готовность модели.
* `/predict` - ожидает запрос, с данными   

Запуск скрипта (predict_scrip.py), который формирует запрос к сервису:
```bash
cd online_inference
pip install -r requirements.txt
python predict_script --in path_to_data_csv_file --host 127.0.0.1 --port 8000
```
Запуск тестов:
```bash
cd online_inference
pip install -r test_requirements.txt
pip install "git+https://github.com/made-ml-in-prod-2022/FiliVan@homework1_last#egg=ml_project&subdirectory=ml_project"
pytest
```
Для уменьшения размера docker-образа:
1. Выбрана легковесная версия базового образа (python3.9:slim);
2. Уменьшено количетсво "слоёв" за счёт объединения команд;
3. Установка пакетов через pip без cache;  
4. Применил мульти-ступенчантую сборку с переносом записимостей.  
**В итоге** обрза занимает 730МБ.  
Чтобы добиться более значительного уменьшения веса docker-образа нужно разделить прошлый проект на predict/train и только predict части. Этого я не делал.