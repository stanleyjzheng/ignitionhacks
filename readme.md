Please see devpost.

The following is my submission for Division Sigma (NLP Tweet classification). This hackathon is a 2 day hackathon, but I have an appointment, so only have about 15 hours. Many corners were cut, but I tried my best to preserve performance.

## Details
- Model: Huggingface roBERTa base pretrainedtrained by JPLU https://huggingface.co/jplu/tf-xlm-roberta-base
- Tokenizer: Autotokenizer using jplu/tf-xlm-roberta-base weights
- "Ensembled" different hyperparameters by averaging CSV's (may or may not make it to final model, check GitHub)
- Cross-validation: Random data from train
- Pseudolabelling 2 folds at 10 epochs (check GitHub to make sure, as I change this frequently)

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

If you have any questions about my submission, please leave them in the comments below or message me on discord at Stanley#1933

main.ipynb includes training as well as inference. It was run on dual Nvidia P100's.

