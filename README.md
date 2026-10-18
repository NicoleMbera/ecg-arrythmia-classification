ECG Heartbeat Arrhythmia Classification: Traditional ML vs Deep Learning

Classifying single ECG heartbeats into the five AAMI classes (N, S, V, F, Q) on the MIT-BIH Arrhythmia dataset, comparing scikit-learn models with Keras networks. Built for the Introduction to Machine Learning summative at ALU.

Links: Report (PDF) · Google Drive (dataset and models) · Video

Results (test set)
Model	Macro-F1	Balanced acc.	Recall S	Recall F
ML-5: tuned gradient boosting	0.901	0.863	0.707	0.698
DL-5: CNN with batch normalisation	0.924	0.939	0.847	0.901

On 22 patients the models had never seen, macro-F1 (N, S, V, F) drops to 0.393 (ML-5) and 0.420 (DL-5).

Files
File	What it is
ecg_arrhythmia_final.ipynb	Main notebook: data exploration, 17 experiments (ML-1 to ML-7, DL-1 to DL-8), test comparison and error analysis
ecg_inter_patient.ipynb	Retrains ML-5 and DL-5 on 22 patients and tests them on 22 unseen patients
experiment_table.csv	Every run with its settings and scores
tensorboard_logs.zip	TensorBoard and CSV logs of the deep-learning runs
report/	The final report
How to run
Open the notebook on Kaggle and add the dataset shayanfazeli/heartbeat with Add Input.
Turn on a GPU (T4) and, for the inter-patient notebook, Internet.
Run all cells. The main notebook takes about an hour on a GPU.

To view the logs: unzip tensorboard_logs.zip and run tensorboard --logdir logs.

Loading the saved models
python
import joblib, tensorflow as tf

ml = joblib.load("best_traditional_model.joblib")   # {"pipeline": ..., "feature_set": "eng+ds"}
dl = tf.keras.models.load_model("best_dl_model.keras")

The traditional model expects engineered features, so use the feature_set() helper from the notebook before calling ml["pipeline"].predict(...). The deep model takes raw beats with shape (n, 187, 1).

Data

MIT-BIH Arrhythmia Database (Moody and Mark, 2001; PhysioNet) in the preprocessed version by Kachuee et al. (2018).
