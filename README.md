# mipt_speech_course
Solutions for course "Speech Technologies"
В задании нужно саггрегировать предсказания от трех моделей с помощью 
* ROVER
* MBR

Модели - conformer-las, conformer-ctc и conformer-wide-ctc.

WER каждой из моделей по отдельности:
* conformer-las: 0.423
* conformer-ctc: 0.425
* conformer-wide-ctc: 0.37

Получившийся WER при аггрегации ROVER-ом:
* невзвешенный ROVER: 0.357
* ROVER с кривым взвешениванием: 0.347

Получившийся WER при аггрегации MBR-ом:
* невзвешенный MBR: 0.352
* взвешенный MBR (веса подобраны эмпирически): 0.344

Какой теоретический максимум можно было вытащить из гипотез:
* Oracle WER: 0.282
