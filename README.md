# Проект по распознаванию печатей и информации
## Описание
Данный проект предназначен для распознавания печатей и извлечения информации, содержащейся в них. Он включает в себя несколько этапов обработки изображений, начиная от предобработки и выравнивания текста до его распознавания с использованием глубоких нейронных сетей.

## Основные функции
### 1. Предобработка изображений:

* Загрузка изображений и их преобразование в черно-белый формат.
* Применение методов морфологической обработки (эрозия и дилатация) для улучшения качества изображений и устранения шумов.
### 2. Выравнивание текста:

* Использование методов интерполяции и кривых для выравнивания кривого текста, что позволяет улучшить качество распознавания.
### 3. Распознавание текста:

* Интеграция с библиотекой MMOCR для обнаружения текста на изображениях.
* Использование предобученных моделей для повышения точности обнаружения.
### 4. Извлечение информации:

* Извлечение и анализ текстовой информации из изображений печатей.
* Генерация метаданных для сохраненных изображений.
### 5. Генерация тестовых данных:

* Возможность создания синтетических изображений печатей с текстом для тестирования и обучения моделей.
### 6. Разделение текста:

* Автоматическое разделение текста на отдельные слова для дальнейшей обработки и анализа.

## Установка
Для запуска проекта необходимо выполнить следующие команды:
conda create -n stamp_recognition python=3.8 pytorch=1.10 cudatoolkit=11.3 torchvision -c pytorch -y
conda activate stamp_recognition
pip3 install openmim
git clone https://github.com/KirillYd/stamp_recognition.git
cd stamp_recognition
mim install -e .
pip install requirements.txt

## Использование
* В файле infer.py на строке 231 можно указать количество генерируемых изображений.
* Для генерации печатей необходимо раскомментировать строку 233 в infer.py
* Чтобы обработать свои печати, подготовьте изображения печатей. Они должны находится в папке dataset/images
* Запустите скрипт, передав путь к изображению или папке с изображениями в качестве аргумента. (python tools/infer.py dataset/test.jpg --det drrg)
* Результаты распознавания будут выведены в отдельном окне.

## Примечания
Для улучшения качества распознавания рекомендуется использовать изображения с высоким разрешением.
