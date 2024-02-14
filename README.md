Plasma Processing Program

Программа разработана для быстрого поиска филаментов в сигнале плазмы. Для первичной фильтрации шума использцется метод Быстрого Преобразования Фурье и расчёт 2-ой производной. 
Обработанные фрагменты фильтруются с помощью нейросети-фильтра (бинарный классификатор).

Работа с программой:
1. Файл с филаментами (расширение .dat) загрузить в область запуска программы (в корень проекта)
2. Для начала работы запустить файл plasma.py и следовать инструкциям
   * После окончания процесса обработки и фильтрации программа выведет график распределения количества отфильтрованных филаментов от поставленной вероятностной границы (с какой вероятностью нейросеть определяет данный фрагмент, как филамент)
   * В программе указан порог 0.75
3. После выполнения архив с филаментами data_tot.zip будет доступен в области запуска программы (в корне проекта)
   * В корне архиве находятся отфильтрованные нейросетью филаменты
   * В папке tot находятся все филаменты выделенные до фильтрации
4. Также программа считает статистику по каждому отфильтрованному филаменту и выгружает её в excel таблицу (в корне проекта)

!Warning!
Проверьте место запуска прграммы, чтобы не было конфликтов в работе програамы и с поиском файлов.