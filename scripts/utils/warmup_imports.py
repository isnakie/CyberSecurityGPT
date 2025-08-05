# warmup_imports.py
print("Starting import warmup...")

import numpy
import pandas
import matplotlib
import seaborn
import sklearn
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

print("All imports completed.")