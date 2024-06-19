# Предиктивная аналитика больших данных

Учебный проект для демонстрации основных этапов жизненного цикла проекта предиктивной аналитики.  

## Installation 

Клонируйте репозиторий, создайте виртуальное окружение, активируйте и установите зависимости:  

```sh
git clone https://github.com/pany1223/pabd24
cd pabd24
python -m venv venv

source venv/bin/activate  # mac or linux
.\venv\Scripts\activate   # windows

pip install -r requirements.txt
```

## Usage

### 1. Сбор данных о ценах на недвижимость 
Собирает данные о ценах на недвижимость однокомнатных, двухкомнатных и трёхкомнатных квартир в Москве с циана.

```
python parse_cian.py
```
 

### 2. Выгрузка данных в хранилище S3 
Передать имя файлов, который надо загрузить. Для доступа к хранилищу скопируйте файл `.env` в корень проекта.  

```
python upload_to_s3.py
```

### 3. Загрузка данных из S3 на локальную машину  
Передать имя файлов, который надо скачать. С
```
python preprocess_data.py
```

### 4. Предварительная обработка данных  
Сначала создать папку log, и папку proc в папку data. Скрипт преобразует спрашенные даныне в датасеты и записывает данные в data/proc.
```
python preprocess_data.py
```

### 5. Обучение модели 
Скрипт обучает модель и сохраняет её. Используется линейная регрессия, где зависимая переменная - цена, независимая - количество метров
```
python train_model.py
```

### 6. Запуск приложения flask 

todo

### 7. Использование сервиса через веб интерфейс 

Для использования сервиса используйте файл `web/index.html`.  
