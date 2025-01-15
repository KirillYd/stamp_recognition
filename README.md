# Проект по распознаванию печатей и информации
## Описание
Данный проект предназначен для распознавания печатей и извлечения информации, содержащейся в них. Он включает в себя несколько этапов обработки изображений, начиная от предобработки и выравнивания текста до его распознавания с использованием глубоких нейронных сетей.

## Основные функции
### 1. Предобработка изображений:

* Загрузка изображений и их преобразование в черно-белый формат.
* Применение методов морфологической обработки (эрозия и дилатация) для улучшения качества изображений и устранения шумов.
### 2. Выравнивание текста:

Использование методов интерполяции и кривых для выравнивания кривого текста, что позволяет улучшить качество распознавания.
### 3. Распознавание текста:

Интеграция с библиотекой MMOCR для обнаружения и распознавания текста на изображениях.
Использование предобученных моделей для повышения точности распознавания.
### 4. Извлечение информации:

Извлечение и анализ текстовой информации из изображений печатей.
Генерация метаданных для сохраненных изображений.
Генерация тестовых данных:

Возможность создания синтетических изображений печатей с текстом для тестирования и обучения моделей.
Разделение текста:

Автоматическое разделение текста на отдельные слова для дальнейшей обработки и анализа.
Установка
Для запуска проекта необходимо установить все зависимости, указанные в requirements.txt. Убедитесь, что у вас установлены следующие библиотеки:

OpenCV
NumPy
Matplotlib
Pygam
SciPy
TensorFlow и другие библиотеки, используемые в проекте.
Использование
Подготовьте изображения печатей, которые вы хотите обработать.
Запустите скрипт, передав путь к изображению или папке с изображениями в качестве аргумента.
Результаты распознавания будут сохранены в указанной директории.
Примечания
Обратите внимание на наличие необходимых шрифтов и изображений, используемых в проекте.
Для улучшения качества распознавания рекомендуется использовать изображения с высоким разрешением.
