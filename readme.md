
## Details
- Model: Huggingface roBERTa base pretrainedtrained by JPLU https://huggingface.co/jplu/tf-xlm-roberta-base
- Tokenizer: Autotokenizer using jplu/tf-xlm-roberta-base weights
- "Ensembled" different hyperparameters by averaging CSV's (may or may not make it to final model, check GitHub)
- Cross-validation: Random data from train
- Pseudolabelling 2 folds at 10/12 epochs (check GitHub to make sure, as I change this frequently)

Training time is about 7 hours on a P100 and about 4-5 hours for inference (due to pseudolabelling, which takes a while).

## To Improve
Due to the time constraints, there are many areas I could/want to improve.
- Try roBERTa large (takes much longer to train, possibly much better performance with such a large dataset. VRAM would be an issue)
- Data stratification 
- K-Folds or group k-folds (Can probably prevent overfitting and offer more accurate validation)
- Customize tokenizer (I've never done this, so I'm not sure what the impact could be)
- Bayesian hyperparameter tuning (Currently tuned with intuition and trial-and-error)
- Save the dataset as a tfrecord so I/O is faster (I/O on Google Colab is painfully slow) for faster prototyping
- LR scheduler (this takes a while to tune, so perhaps more suited to a long term competition)
- Preprocess data better
- Data visualization to build more intuition for hparam tuning

## What I am proud of
- Rapid prototyping allowed by having multiple sessions. 
- Relatively effective, quick cross-validation
- Pseudolabelling in 16 hours! I'm super proud of this, and it's probably the reason my model performs so well.

Thanks for reading, and please check out the Github link below. If you do data science competitions, please follow me on Kaggle. I'd love to work together - https://www.kaggle.com/stanleyjzheng

## Aug 30 documentation update

First of all, thanks to Dev for evaluating my tons of CSV's so that I can understand what makes them work. I did a ton of experimenting throughout the competition, so naturally, I have quite a few CSV's. Out of luck, I happened to submit the CSV that performed the best. Here is what helped me the most.

1. Model selection. This is quite important. Looking at [Papers with Code's Leaderboard](https://paperswithcode.com/sota/sentiment-analysis-on-sst-2-binary), we see many extremely large models using extra training data. The one that stood out to me was RoBERTa, with amazing accuracy, yet is possible to train on excellent hardware in our timeframe. That's how I chose RoBERTa for this application. In this particular dataset, SST-2, models are used to classify binary sentiments similar to our competition. However, this is a handlabelled dataset, while our dataset was machine labelled. This will come into play in point #4 and #5. The XLM-trained model I used contains updated pretrained weights for roBERTa. In every metric, it outperforms roBERTa while being identical in computation requirements. I encourage you to read the [ArXiv](https://arxiv.org/pdf/1911.02116.pdf). In addition, an ensemble would no doubt have done better, but given the timeframe, this isn't possible. If I had more time, I would have liked to experiment with XLnet. In the benchmark above, XLnet performs 0.3% superior to roBERTa. However, it XLM-roBERTa may be very similar in performance.

2. A great baseline. I have built up intuition for hyperparameters, and alongside some bayesian statistics, hyperparameter tuning is much easier. A good baseline is extremely important, as pseudolabelling is exponentially more effective if you have a better baseline. I spent a few hours tuning hyperparameters before letting my model train overnight. Looking back, seeing the score of my baseline, I believe it is actually underfit (a score of 91-ish should be possible), but it may have actually helped my pseudolabelling.

3. Pseudolabelling. You've probably heard me talking about this and recommending it to others. Essentially, pseudolabelling is labelling the test set with your model, then training your model on the test set. One fold means that your model is trained on the training data, infers the test set, trains on the test set, then infers one more time to create a submission. Two folds means that instead of creating a submission, it trains on the second test set a second time, and then makes a submission. (Note that the name 1 folds and 2 folds sounds like k-folds validation, it's a bit like k-folds but without validation. I would have used k-folds in this scenario, but I don't think it would've helped very much.)

If you would like to learn more about pseudolabelling for computer vision, please check out my [Global Wheat Competition Notebook](https://www.kaggle.com/stanleyjzheng/yolov4pl-oof). It helped me get 75th out of 2245 teams in a Kaggle object detection competition, earning me my first medal. It is pretty elegant code written in Pytorch. Before pseudolabelling, my score was 0.70, after was 0.745. Keep in mind, the test set was hand labelled, not machine labelled, so a more general model does better in this case; even so, pseudolabelling boosts the score significantly. [Here is a video](https://www.youtube.com/watch?v=SsnWM1xWDu4) that I love, by a Kaggle grandmaster on his pseudolabelling solution. This is the proper way to do it, but practically impossible given our timeframe.

4. Machine-made data. In Kaggle competitions, your training set is normally machine labelled, while your test set is human labelled. This way, your data preprocessing is crucial; you are trying to make the most generalized model so that it performs well on more accurate, human labelled data. However, this dataset is completely different. It is completely machine labelled, meaning you're not trying to find the most generalized solution, but the solution most similar to the model that created the dataset. Therefore, data preprocessing that is similar to the original data is important. I didn't have time to experiment, so I didn't bother with data preprocessing, but good data preprocessing could've won this competition easily.

5. Overfit. You probably avoid overfitting, but there's many cases where overfitting helps. As I said, this competition differs from Kaggle competitions since its data is completely machine labelled. I was careful not to overfit too much to the training data, stopping training when my relatively small 50000 long validation set's loss crossed over with my training set (well, a bit further, but close enough). I then relied on pseudolabels. It's usually quite difficult to overfit to pseudolabels, so I decided to go with 2 folds. First fold was 12 epochs, and second fold was 10 epochs.

Truth be told, I've never done 2 folds with a model before; Kaggle requires final submissions take less than 6 hours on an Nvidia Tesla P100 (so most of my past submissions would run out of time for 2 folds), while mine took more than 4 hours on 2x Nvidia Tesla P100. Most of these steps were luck; I got lucky 2 folds worked better than 1, and that pseudolabelling worked at all. I fully admit that luck played a large part in my scores. If the final test data had been hand-labelled, I would no doubt have gotten under 80%. These are just things you have to deal with when it comes to competitive data science; knowing what your data is made up of is extremely important.

I am extremely happy with my score and just wanted to share a few of the reasons for it. Thanks for being such a great community, I had a ton of fun! I wish everyone who participated in this competition the best of luck, and please feel free to message me on Discord at Stanley#1933 if you have any questions. I will put any FAQ below.
