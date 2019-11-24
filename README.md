# NeyroWineQuality
Материалы тестового задания: 
построить модель определения качества вина на основе его физико-химических характеристик.

## Инструкция по запуску
### Чтобы запустить ноутбук с семинара на своей машине:

1) Cклонируйте репозиторий курса:

`git clone https://github.com/KrisAnTis-Group/NeyroWineQuality.git`

2) В терминале выполните команду:

`pip install -r requirements.txt`

3) Запустите один из скриптов *.py:

`Dense_red.py`
`first_vine_white.py`
`RedAndWhite.py`

#### Замечание: Результатом работы считается файл RedAndWhite - он объединяет white и red наборы данных

### Чтобы запустить ноутбук на [Google Colab](https://colab.research.google.com):

0) Откройте [Google Colab](https://colab.research.google.com)

1) Скачайте ноутбук (вкладка Github), затем прописываете адрес репозитория.

2) Чтобы выкачать на colab библиотеку dlnlputils, не забудьте выполнить команду в первой ячейке:

...
import sys; sys.path.append('/content/NeyroWineQuality')
!git clone https://github.com/KrisAnTis-Group/NeyroWineQuality.git && pip install -r NeyroWineQuality/requirements.txt
...

3) Не забудьте настроить `device='cpu'` (модель работает на cpu - установлено по умолчанию на Google Colab), а также выбрать подходящий Runtime в Google Colab (CPU/TPU/GPU).

4) Запустите ноутбук.
## DataSet
[Wine Quality Data Set](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)
  
Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.
