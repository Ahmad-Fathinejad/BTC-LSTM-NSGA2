# -*- coding: utf-8 -*-
import time
import numpy as np
import json
import logging # اضافه شد
from deap import algorithms, tools
from src import config, data_preparation, model_optimizer

def main():
    # --- تنظیمات لاگ‌گیری ---
    # تمام خروجی‌ها هم در کنسول و هم در فایل optimization.log ذخیره می‌شوند
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("optimization.log", mode='w'), # ذخیره در فایل
            logging.StreamHandler() # نمایش در کنسول
        ]
    )
    # --- پایان تنظیمات لاگ‌گیری ---

    logging.info("Log recording started.") # جایگزین print

    # # تابع اصلی برای اجرای کل پایپ‌لاین بهینه‌سازی.
    # # Main function to run the entire optimization pipeline.
    # # آماده‌سازی داده‌ها
    # # Prepare the data
    scaled_data, scaler, feature_df = data_preparation.load_and_prepare_data()

    # # تنظیم متغیرهای گلوبال برای استفاده در تابع شایستگی
    # # Set global variables for use in the fitness function
    model_optimizer.scaled_data = scaled_data
    model_optimizer.feature_df = feature_df
    # # پیکربندی جعبه‌ابزار NSGA-II
    # # Configure the NSGA-II toolbox
    toolbox = model_optimizer.setup_nsga2_toolbox()
    # # پارامترهای الگوریتم
    # # Algorithm parameters
    NGEN = config.NGEN
    POP_SIZE = config.POP_SIZE
    CXPB = config.CXPB
    MUTPB = config.MUTPB
    
    logging.info("\n" + "="*50) # جایگزین print
    logging.info(" 🚀  Starting NSGA-II Optimization...") # جایگزین print
    # # شروع فرآیند بهینه‌سازی با الگوریتم NSGA-II...
    logging.info(f"Generations: {NGEN}, Population Size: {POP_SIZE}") # جایگزین print
    logging.info("="*50) # جایگزین print
    
    # # ایجاد جمعیت اولیه
    # # Create initial population
    population = toolbox.population(n=POP_SIZE)

    # # برای نگهداری بهترین راه‌حل‌های غیرمغلوب (جبهه پارتو)
    # # To store the best non-dominated solutions (Pareto front)
    hall_of_fame = tools.ParetoFront()
    # # تعریف آمار برای نمایش
    # # Define statistics for display
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("min", np.min, axis=0)

    # # شروع اندازه‌گیری زمان
    # # Start timing
    start_time = time.time()

    # # اجرای الگوریتم
    # # Run the algorithm
    algorithms.eaMuPlusLambda(
        population, toolbox, mu=POP_SIZE, lambda_=POP_SIZE,
        cxpb=CXPB, mutpb=MUTPB, ngen=NGEN,
        stats=stats, halloffame=hall_of_fame, verbose=True
    )

    # # پایان اندازه‌گیری زمان
    # # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f" ⏱️  Total optimization time: {elapsed_time/60:.2f} minutes.") # جایگزین print
    
    # # نمایش نتایج
    # # Display results
    logging.info("\n" + "="*50) # جایگزین print
    logging.info(" 🏆  Pareto Front (Best Solutions Found)  🏆 ") # جایگزین print
    # # جبهه پارتو (بهترین راه‌حل‌های یافت‌شده)
    logging.info("="*50) # جایگزین print

    for individual in hall_of_fame:
        params = {
            "validation_loss": individual.fitness.values[0],
            "num_parameters": individual.fitness.values[1],
            "hyperparameters": {
                "timesteps": int(individual[0]),
                "lstm_units": [int(individual[1]), int(individual[3])],
                "dropout": [round(individual[2], 2), round(individual[4], 2)],
                "batch_size": int(individual[5]),
                "patience": int(individual[6])
            }
        }
        logging.info(json.dumps(params, indent=2)) # جایگزین print
        logging.info("-" * 20) # جایگزین print

if __name__ == "__main__":
    main()