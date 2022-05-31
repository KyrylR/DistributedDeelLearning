# Distributed Deep Learning (Server part)

Development of the COVID-19 Disease Prediction and Analysis Application Using Distributed Computing

## Description

На серверній частині застосунку зображення обробляється з таким алгоритмом:
1.	Валідація зображення

Перевіряється чи дійсно було надіслано зображення, якщо ні, то видається відповідна помилка.

2.	Збереження зображення на сервері
3.	Перед обробка зображення
4.	Оцінка вхідного зображення нейронної моделлю

При першому надсиланні запиту на сервер, модель буде завантажена в пам’ять, тому перший запит довший, ніж попередні. 

5.	Відправка отриманого результату на клієнт


### Reference Documentation

For further reference, please consider the following sections:
[1.	Tom M. Mitchell Machine Learning ](https://archive.org/details/machinelearning0000mitc)
[2.	Машинне навчання ](https://en.wikipedia.org/wiki/Artificial_neural_network)
[3.	Штучна нейронна мережа ](https://en.wikipedia.org/wiki/Machine_learning)
[4.	Deep Learning ](http://www.deeplearningbook.org)
[5.	Chest X-Ray Images (Pneumonia) ](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia.)
[6.	Згорткова нейронна мережа ](https://en.wikipedia.org/wiki/Convolutional_neural_network.)

## Authors

ex. Kyrylo Riabov  
ex. [Gmail](kyryl.ryabov@gmail.com)

## License

This project is licensed under the [MIT] License - see the LICENSE.md file for details

