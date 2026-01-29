# -*- coding: utf-8 -*-
import random
import numpy as np
import logging # اضافه شد
from deap import base, creator, tools
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from src import config
# # این متغیرها به صورت گلوبال تعریف می‌شوند تا در تابع شایستگی قابل دسترس باشند
# # These variables are defined globally to be accessible in the fitness function
scaled_data = None
feature_df = None
def setup_nsga2_toolbox():
    # # جعبه‌ابزار الگوریتم ژنتیک را برای NSGA-II پیکربندی می‌کند.
    # # Configures the Genetic Algorithm toolbox for NSGA-II.
    # # تعریف شایستگی دوهدفه (هر دو هدف کمینه می‌شوند)
    # # Define the two-objective fitness (minimizing both)
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
    # # تعریف ساختار هر فرد (کروموزوم)
    # # Define the structure of an individual
    creator.create("Individual", list, fitness=creator.FitnessMulti)
    toolbox = base.Toolbox()
    # # ثبت ژن‌ها (هر هایپاپارامتر و محدوده آن)
    # # Register the genes (each hyperparameter and its range)
    hp = config.HYPERPARAMETERS
    toolbox.register("attr_timesteps", random.choice, hp['timesteps'])
    toolbox.register("attr_lstm_units_1", random.choice, hp['lstm_units_1'])
    toolbox.register("attr_dropout_1", random.uniform, hp['dropout_1'][0], hp['dropout_1'][1])
    toolbox.register("attr_lstm_units_2", random.choice, hp['lstm_units_2'])
    toolbox.register("attr_dropout_2", random.uniform, hp['dropout_2'][0], hp['dropout_2'][1])
    toolbox.register("attr_batch_size", random.choice, hp['batch_size'])
    toolbox.register("attr_patience", random.choice, hp['patience'])
    # # تعریف ساختار کروموزوم
    # # Define the chromosome structure
    attributes = (
        toolbox.attr_timesteps, toolbox.attr_lstm_units_1, toolbox.attr_dropout_1,
        toolbox.attr_lstm_units_2, toolbox.attr_dropout_2, toolbox.attr_batch_size,
        toolbox.attr_patience
    )
    toolbox.register("individual", tools.initCycle, creator.Individual, attributes, n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    # # ثبت کردن توابع اصلی الگوریتم
    # # Register the core algorithm operators
    toolbox.register("evaluate", evaluate_model)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
    toolbox.register("select", tools.selNSGA2)

    return toolbox
def evaluate_model(individual):
    # # تابع شایستگی: یک فرد را ارزیابی کرده و امتیازات دوهدفه آن را برمی‌گرداند.
    # # Fitness function: evaluates an individual and returns its two-objective scores.
    try:
        # # ۱. خواندن هایپاپارامترها
        # # 1. Unpack hyperparameters
        timesteps, lstm_units_1, dropout_1, lstm_units_2, dropout_2, batch_size, patience = individual
        # # ۲. ایجاد دنباله‌ها بر اساس مقدار timesteps
        # # 2. Create sequences based on the timesteps value
        X_seq, y_seq = [], []
        close_idx = feature_df.columns.get_loc(config.TARGET_COLUMN)
        for i in range(timesteps, len(scaled_data)):
            X_seq.append(scaled_data[i-timesteps:i])
            y_seq.append(scaled_data[i, close_idx])
        X_seq, y_seq = np.array(X_seq), np.array(y_seq)

        # # ۳. تقسیم داده‌ها به آموزشی و ارزیابی
        # # 3. Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=0.25, shuffle=False)

        if len(X_train) == 0 or len(X_val) == 0: return (999.0, 9999999.0)
        # # ۴. ساخت مدل
        # # 4. Build the model
        model = Sequential([
            LSTM(units=int(lstm_units_1), return_sequences=True, input_shape=(timesteps, X_train.shape[2])),
            Dropout(dropout_1),
            LSTM(units=int(lstm_units_2), return_sequences=False),
            Dropout(dropout_2),
            Dense(units=1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')

        num_params = model.count_params()

        # # ۵. آموزش مدل
        # # 5. Train the model
        early_stopping = EarlyStopping(monitor='val_loss', patience=int(patience), restore_best_weights=True)
        history = model.fit(X_train, y_train, batch_size=int(batch_size), epochs=50,
                            validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=0)

        min_val_loss = min(history.history['val_loss'])

        return (min_val_loss, float(num_params))

    except Exception as e:
        logging.error(f"An error occurred for individual {individual}: {e}") # جایگزین print
        return (999.0, 9999999.0)