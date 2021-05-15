# kaggle-02-disaster-tweets

I took part in Kaggle's NLP competition called Disaster Tweets. The goal was to predict if given tweet is about real catastrophe or not.

https://www.kaggle.com/c/nlp-getting-started/overview

The best result was obtained using BERT. At the moment of writing it allowed me to take 335th place out of 2698 participants. 

This actual setup was not much tuned so there is for sure possibility to get higher scores. As the training set was really tiny, the risk of overfitting and not finding patterns during training was very high. I saw that others were pre-processing almost on single tweet basis, which I saw as not the best way of spending my time :) I also didn't want to copy others work just to get higher score.

Best results for algorithms tested:

| Algorithm     | Score        |
| ------------- |:------------:|
| BERT          | 0.83604      | 
| Small BERT    | 0.82470      |
| MLP with avg  | 0.81029      |
| LSTM          | 0.80815      |
| SVM           | 0.80539      |
| Log Reg       | 0.78976      |

To see detailed configurations explore notebooks. You will find the best outcomes and actual code used to obtain them.

For more work-related stuff you can find me on [Twitter](https://twitter.com/MikolajKania). 