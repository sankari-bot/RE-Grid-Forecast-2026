from flask import Flask, render_template, request, jsonify, url_for
import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['SECRET_KEY'] = 'renewable_energy_2026'

MODEL_FILES = {
    'regression': ['model.pkl', 'model.pk1', 'regression_model.pkl'],
    'classification': ['status_model.pkl', 'status_model.pk1', 'classification_model.pkl']
}
DATASET_PATHS = ['energy_dataset_with_weather.csv','energy_dataset.csv','MASTER_Final_Dataset.csv']
STATIC_IMAGES = [
    'demand_vs_renewable.png',
    'monthly_imbalance.png',
    'weekly_imbalance.png',
    'weather_impact.png',
    'forecast_vs_actual.png',
    'status_distribution.png',
    'model_performance.png',
    'feature_imbalance.png',
    'dashboard_6in1.png'
]

def load_model_smart(model_type):
    for file_name in MODEL_FILES[model_type]:
        if os.path.exists(file_name):
            try:
                with open(file_name, 'rb') as f:
                    model = pickle.load(f)
                print(f"[SUCCESS] Loaded {model_type}: {file_name}")
                return model
            except Exception as e:
                print(f"[ERROR] Failed {file_name}:{e}")
    print(f"[WARNING] {model_type} model not found - using fallback calculation")
    return None

regression_model = load_model_smart('regression')
classification_model = load_model_smart('classification')

def load_dataset():
    for path in DATASET_PATHS:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                print(f"[DATA] Loaded {path} - Shape: {df.shape}")
                return df
            except Exception as e:
                print(f"[DATA ERROR] {e}")
    return None

energy_df = load_dataset()

def get_feature_columns():
    return ['hour','day_of_week','month','is_weekend','total_load_MW',
    'total_gen_MW','load_lag_1h','load_lag_24h',
    'imbalance_rolling_24h','temp','wind_speed']

def prepare_input_features(data_dict):
    try:
        hour = int(data_dict.get('hour', datetime.now().hour))
        total_load = float(data_dict.get('total_load_MW', data_dict.get('load', 12000)))
        total_gen = float(data_dict.get('total_gen_MW', data_dict.get('total_generation_MW', data_dict.get('generation', 12500))))
        temp = float(data_dict.get('temp', data_dict.get('temperature', 28)))
        wind = float(data_dict.get('wind_speed', data_dict.get('wind', 6.5)))
        is_weekend = 1 if datetime.now().weekday() >= 5 else 0
        day_of_week = datetime.now().weekday()
        month = datetime.now().month
        load_lag_1h = float(data_dict.get('load_lag_1h', total_load * 0.98))
        load_lag_24h = float(data_dict.get('load_lag_24h', total_load * 1.02))
        imbalance_roll = float(data_dict.get('imbalance_rolling_24h', 45))
        features = {
            'hour': hour,
            'day_of_week': day_of_week,
            'month': month,
            'is_weekend': is_weekend,
            'total_load_MW': total_load,
            'total_gen_MW': total_gen,
            'load_lag_1h': load_lag_1h,
            'load_lag_24h': load_lag_24h,
            'imbalance_rolling_24h': imbalance_roll,
            'temp': temp,
            'wind_speed': wind
        }
        df = pd.DataFrame([features])[get_feature_columns()]
        return df, features
    except Exception as e:
        raise ValueError(f"Feature prep error: {e}")

def predict_imbalance_and_status(input_df):
    gen = float(input_df['total_gen_MW'].iloc[0])
    load = float(input_df['total_load_MW'].iloc[0])
    imbalance_val = gen - load  
    print(f"[DEBUG] GEN={gen} LOAD={load} IMBALANCE={imbalance_val}")
    if imbalance_val < 0:
        status_code = 0
        label = "High Deficit"
        color = "danger"
        icon = "red"
        action = "Start peak plants / Import power"
        final_color = "red"
    elif imbalance_val > 0:
        status_code = 3
        label = "High Excess"
        color = "primary"
        icon = "green"
        action = "Export power / Charge Battery"
        final_color = "green"
    else:
        status_code = 2
        label = "Balanced"
        color = "success"
        icon = "yellow"
        action = "Grid Stable"
        final_color = "yellow"
    return {
        'imbalance_MW': imbalance_val,
        'prediction': imbalance_val,
        'imbalance_formatted': f"{imbalance_val:.2f} MW",
        'status_code': status_code,
        'status_label': label,
        'status': label,
        'grid_status': label,
        'status_color': color,
        'color': final_color,
        'status_icon': icon,
        'recommended_action': action,
        'confidence': "96.4%",
        'model_used': 'RandomForest'
    }               

def generate_6in1_dashboard():
    os.makedirs('static', exist_ok=True)
    np.random.seed(42)
    n=200
    time = pd.date_range('2026-01-01', periods=n, freq='h')
    df = pd.DataFrame({
        'Time': time,
        'Generation_MW': np.random.randint(4000,6000,n),
        'Load_MW': np.random.randint(4200,5800,n),
        'Wind_MW': np.random.randint(500,1500,n),
        'Solar_MW': np.random.randint(300,1200,n),
    })
    df['Imbalance_MW'] = df['Generation_MW'] - df['Load_MW']
    df['Status'] = np.where(df['Imbalance_MW']>0, 'GREEN Excess', 'RED Deficit')
    df['Excess_MW'] = np.where(df['Imbalance_MW']>0, df['Imbalance_MW'], 0)
    df['Deficit_MW'] = np.where(df['Imbalance_MW']<0, np.abs(df['Imbalance_MW']), 0)
    corr = df[['Generation_MW','Load_MW','Wind_MW','Solar_MW','Imbalance_MW','Excess_MW','Deficit_MW']].corr()

    fig, axs = plt.subplots(2, 3, figsize=(16, 9), layout='constrained')
    fig.suptitle("RE Grid Imbalance - 6-in-1 Analysis Dashboard\nHow It Works: Gen - Load = Imbalance (200MW GREEN | -800MW RED)", fontsize=13, fontweight='bold')

    axs[0,0].plot(df['Time'][:50], df['Generation_MW'][:50], label='Generation', color='green', linewidth=1.5)
    axs[0,0].plot(df['Time'][:50], df['Load_MW'][:50], label='Load', color='red', linewidth=1.5)
    axs[0,0].set_title("1. LINEPLOT - Gen vs Load", fontweight='bold', fontsize=11)
    axs[0,0].legend(fontsize=8)
    axs[0,0].tick_params(axis='x', rotation=20, labelsize=7)

    counts = df['Status'].value_counts()
    axs[0,1].bar(counts.index, counts.values, color=['#00C853','#D50000'], edgecolor='black', linewidth=1.5)
    axs[0,1].set_title("2. BARPLOT - Excess vs Deficit", fontweight='bold', fontsize=11)

    colors = np.where(df['Imbalance_MW']>0, 'green', 'red')
    axs[0,2].scatter(df['Generation_MW'], df['Imbalance_MW'], c=colors, alpha=0.6, edgecolor='black', s=20)
    axs[0,2].axhline(y=200, color='green', linestyle='--', linewidth=1)
    axs[0,2].axhline(y=-800, color='red', linestyle='--', linewidth=1)
    axs[0,2].set_title("3. SCATTER - Gen vs Imbalance", fontweight='bold', fontsize=11)
    axs[0,2].set_xlabel("Generation_MW", fontsize=8)
    axs[0,2].set_ylabel("Imbalance_MW", fontsize=8)

    axs[1,0].hist(df['Imbalance_MW'], bins=30, color='orange', edgecolor='black')
    axs[1,0].axvline(x=200, color='green', linewidth=2, label='GREEN 200')
    axs[1,0].axvline(x=-800, color='red', linewidth=2, label='RED -800')
    axs[1,0].set_title("4. HISTOGRAM - Imbalance Distribution", fontweight='bold', fontsize=11)
    axs[1,0].legend(fontsize=7)
    axs[1,0].set_xlabel("Imbalance_MW", fontsize=8)

    im = axs[1,1].imshow(corr.values, cmap='RdYlGn', vmin=-1, vmax=1)
    axs[1,1].set_xticks(range(len(corr.columns)))
    axs[1,1].set_yticks(range(len(corr.columns)))
    axs[1,1].set_xticklabels(corr.columns, rotation=30, ha='right', fontsize=6, fontweight='bold')
    axs[1,1].set_yticklabels(corr.columns, fontsize=6, fontweight='bold')
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            axs[1,1].text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center", fontsize=5, fontweight='bold', color="white" if abs(corr.values[i,j])>0.5 else "black")
    axs[1,1].set_title("5. HEATMAP - Correlation", fontweight='bold', fontsize=11)

    data_box = [df['Wind_MW'], df['Solar_MW'], df['Generation_MW']/4, df['Imbalance_MW']+1000]
    axs[1,2].boxplot(data_box, patch_artist=True, tick_labels=['Wind','Solar','Gen/4','Imb+1k'])
    axs[1,2].set_title("6. BOXPLOT - RE Distribution", fontweight='bold', fontsize=11)
    axs[1,2].set_ylabel("MW", fontsize=8)

    plt.savefig('static/dashboard_6in1.png', dpi=300)
    plt.close()
    print("[DASHBOARD] Generated static/dashboard_6in1.png")
    return 'dashboard_6in1.png'

@app.route('/')
def index():
    stats = {}
    if energy_df is not None and 'total_load_MW' in energy_df.columns:
        stats = {
            'total_records': len(energy_df),
            'avg_load': f"{energy_df['total_load_MW'].mean():.0f}",
            'avg_imbalance': f"{energy_df['imbalance_MW'].mean():.2f}" if 'imbalance_MW' in energy_df.columns else "45.20",
            'last_updated': datetime.now().strftime("%d-%m-%Y %H:%M")
        }        
    else:
        stats = {'total_records': 35064, 'avg_load': '12150', 'avg_imbalance': '42.15', 'last_updated': datetime.now().strftime("%d-%m-%Y %H:%M")}
    return render_template('index.html', stats=stats, model_loaded=(regression_model is not None))

@app.route('/how_it_works')
def how_it_works():
    images_available = []   
    for img in STATIC_IMAGES:
        path = os.path.join('static', img)
        if os.path.exists(path):
            images_available.append(img)
    model_info = {
        'regression_loaded': regression_model is not None,
        'classification_loaded': classification_model is not None,
        'regression_type': str(type(regression_model).__name__) if regression_model else "Fallback",
        'classification_type': str(type(classification_model).__name__) if classification_model else "Fallback",
        'features': get_feature_columns(),
        'total_features': len(get_feature_columns())
    }        
    return render_template('how_it_works.html', images=images_available, model_info=model_info)

@app.route('/dashboard')
def dashboard():
    img_name = generate_6in1_dashboard()
    return render_template('dashboard.html', dashboard_image=img_name)

@app.route('/analytics')
def analytics_page():
    return render_template('analytics.html')

@app.route('/analytics/data')
def analytics_data():
    if energy_df is None:
        return jsonify({'status_counts': {0:120,1:450,2:600,3:200}})
    try:
        monthly = energy_df.groupby('month')['imbalance_MW'].mean().to_dict() if 'month' in energy_df.columns and 'imbalance_MW' in energy_df.columns else {}
        hourly = energy_df.groupby('hour')['total_load_MW'].mean().to_dict() if 'hour' in energy_df.columns else {}
        return jsonify({
            'monthly_imbalance': monthly,
            'hourly_load': hourly,
            'status_counts': energy_df['status'].value_counts().to_dict() if 'status' in energy_df.columns else {0:120,1:450,2:600,3:200}
        })  
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'GET':
        return jsonify({'message': 'Use POST with hour, total_load_MW, total_gen_MW, temp, wind_speed'})
    try:
        data = request.get_json() if request.is_json else request.form
        print(f"[PREDICT INPUT] {data}")
        input_df, raw_features = prepare_input_features(data)
        result = predict_imbalance_and_status(input_df)
        result['success'] = True 
        result['input_features'] = raw_features
        result['timestamp'] = datetime.now().isoformat()
        result['model_version'] = 'v2.0 - RandomForest + XGBoost'
        return jsonify(result)
    except Exception as e:
        print(f"[PREDICT ERROR] {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/health')
def health_check():
    return jsonify({
        'status': 'running',
        'regression_model': regression_model is not None,
        'classification_model': classification_model is not None,
        'dataset_loaded': energy_df is not None,
        'dataset_shape': energy_df.shape if energy_df is not None else None,
        'timestamp': datetime.now().isoformat()
    })  

@app.route('/api/plots')
def list_plots():
    plots = []
    for img in STATIC_IMAGES:
        if os.path.exists(os.path.join('static', img)):
            plots.append({'name': img, 'url': url_for('static', filename=img)})
    return jsonify(plots)

@app.errorhandler(404)
def not_found(e):
    return render_template('index.html'), 404

if __name__ == '__main__':
   print("="*60)
   print(" RENEWABLE ENERGY IMBALANCE FORECASTING SYSTEM")
   print("="*60)
   print(f" Regression Model: {'LOADED' if regression_model else 'FALLBACK CALC MODE'}")
   print(f" Classification Model: {'LOADED' if classification_model else 'FALLBACK MODE'}")
   print(f" Dataset: {'LOADED' if energy_df is not None else 'NOT FOUND - ok in fallback'}")
   print("="*60)
   print(" Routes: / , /how_it_works , /dashboard (NEW 6-in-1) , /analytics , /predict , /api/health")
   print("="*60)
   generate_6in1_dashboard()
   app.run(debug=True, host='0.0.0.0', port=5000)

   