# IgnitionHacks 2020 Winning Submission

## Details
- Model: Huggingface roBERTa base pretrainedtrained by [JPLU](https://huggingface.co/jplu/tf-xlm-roberta-base), see [paper](https://arxiv.org/abs/1911.02116)
- Tokenizer: Autotokenizer using jplu/tf-xlm-roberta-base weights, see paper above
- Validation: Random data from train
- Pseudolabelling 2 rounds at 10/12 epochs 

Training time is about 7 hours on a P100 and about 4-5 hours for inference (due to pseudolabelling, which takes a while).

## To Improve
Due to the time constraints, there are many areas I could/want to improve.
- Try roBERTa large (takes much longer to train, possibly much better performance with such a large dataset. VRAM would be an issue)
- Data stratification 
- K-Folds or group k-folds (Can probably prevent overfitting and offer more accurate validation)
- Customize tokenizer (I've never done this, so I'm not sure what the impact could be)
- Bayesian hyperparameter tuning (Currently tuned with intuition and trial-and-error)
- Save the dataset as a tfrecord so I/O is faster (I/O on Google Colab is painfully slow) for faster prototyping. A bit of a non-issue since I did all my work within the 24-hour runtime limit, but for a longer hackathon, this is pracrically required.
- LR scheduler (this takes a while to tune, so perhaps more suited to a long term competition)
- Preprocess data better
- Data visualization to build more intuition for hparam tuning

## What I am proud of
- Rapid prototyping allowed by having multiple sessions. 
- Relatively effective, quick cross-validation
- Pseudolabelling in 16 hours
