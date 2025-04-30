### Основное

Для обучения взял модель [DIFM](https://www.ijcai.org/Proceedings/2020/0434.pdf), которая реализована в библиотеки [DeepCTR-Torch](https://github.com/shenweichen/DeepCTR-Torch?tab=readme-ov-file)

Модель имеет следующие особенности:
- Bit-wise: Использует DNN для учёта влияния отдельных элементов эмбеддингов
- Vector-wise: Применяет Multi-Head Self-Attention для анализа взаимодействий между признаками
- Объединяет FM (Factorization Machines), Residual Networks, Self-Attention и DNN в единую модель

### Обучение


