# -*- coding: utf-8 -*-
# --- تنظیمات فایل و داده ---
# --- File and Data Settings ---
DATA_FILE_PATH = "data/btc_usd_1h_from_20230101_to_20250705.csv"
# # نام ستون تاریخ در فایل CSV
# # Name of the date column in the CSV file
DATE_COLUMN = 'date'
# # نام ستون هدف برای پیش‌بینی
# # Name of the target column for prediction
TARGET_COLUMN = 'close'
# --- پارامترهای الگوریتم ژنتیک (NSGA-II) ---
# --- Genetic Algorithm (NSGA-II) Parameters ---
# # تعداد نسل‌ها
# # Number of generations
NGEN = 10
# # اندازه جمعیت در هر نسل
# # Population size in each generation
POP_SIZE = 10
# # احتمال ترکیب (Crossover)
# # Crossover probability
CXPB = 0.7
# # احتمال جهش (Mutation)
# # Mutation probability
MUTPB = 0.2
# --- فضای جستجوی هایپاپارامترها ---
# --- Hyperparameter Search Space ---
HYPERPARAMETERS = {
"timesteps": [30, 60, 90],
"lstm_units_1": [32, 50, 64],
"dropout_1": [0.2, 0.5],
"lstm_units_2": [32, 50, 64],
"dropout_2": [0.2, 0.5],
"batch_size": [32, 64],
"patience": [5, 10]
}