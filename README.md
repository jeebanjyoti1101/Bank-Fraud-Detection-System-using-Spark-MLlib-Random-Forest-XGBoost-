# Fraud Detection System

A machine learning system for detecting fraudulent transactions using scikit-learn.

## 🚀 Quick Start

### Run the Application
```bash
run_app.bat
```
Then open your browser to: **http://localhost:5000**

### Train New Model
```bash
python scripts/train.py
```

## 📁 Project Structure

```
fraud-detection/
├── app/
│   ├── app.py              # Main Flask application
│   ├── static/             # CSS, JS files
│   └── templates/          # HTML templates
├── data/
│   └── Fraud.csv           # Training dataset
├── models/
│   ├── rf_model.pkl        # Trained Random Forest model
│   ├── scaler.pkl          # Feature scaler
│   ├── encoders.pkl        # Categorical encoders
│   └── feature_info.json   # Model metadata
├── scripts/
│   ├── train.py           # Model training script
│   ├── evaluate.py        # Model evaluation utilities
│   └── preprocess.py      # Data preprocessing utilities
└── notebooks/
    └── exploration.ipynb   # Data exploration notebook
```

## 🎯 Model Performance
- **Algorithm**: Random Forest Classifier
- **Accuracy**: 86.4%
- **Features**: 11 engineered features
- **Training Data**: 104,857 samples (16.9% fraud rate)

## 🛠 Technical Details
- **Backend**: Flask + scikit-learn
- **Frontend**: HTML + CSS + JavaScript
- **Features**: Amount patterns, balance analysis, transaction types
- **Deployment**: Local development server

## 📊 Usage
1. Start the application using `run_app.bat`
2. Open http://localhost:5000 in your browser
3. Enter transaction details in the form
4. Get real-time fraud prediction with confidence score

## 🔧 Development
- **Train Model**: `python scripts/train.py`
- **Run App**: `python app/app.py`
- **Requirements**: See `requirements.txt`

## 🌐 Web Application

### Running the Flask App

1. **Start the Application**:
   ```bash
   cd app
   python app.py
   ```

2. **Access the Web Interface**:
   - Open your browser to `http://localhost:5000`
   - The application provides an intuitive interface for fraud detection

### API Endpoints

#### Single Prediction
```bash
POST /predict
Content-Type: application/json

{
  "Time": 0,
  "V1": -1.359807,
  "V2": -0.072781,
  ...
  "V28": -0.021053,
  "Amount": 149.62
}
```

#### Batch Prediction
```bash
POST /predict-batch
Content-Type: application/json

{
  "transactions": [
    {
      "Time": 0,
      "V1": -1.359807,
      ...
      "Amount": 149.62
    },
    ...
  ]
}
```

#### Model Information
```bash
GET /model-info
```

#### Health Check
```bash
GET /health
```

## 📊 Dataset

The system uses the Credit Card Fraud Detection dataset with the following features:

- **Time**: Number of seconds elapsed between this transaction and the first transaction
- **V1-V28**: Principal components obtained with PCA (anonymized features)
- **Amount**: Transaction amount
- **Class**: Response variable (1 for fraud, 0 for normal)

## 🤖 Model Performance

### Random Forest
- **AUC**: ~0.98
- **Accuracy**: ~99.9%
- **F1-Score**: ~0.85

### XGBoost
- **AUC**: ~0.99
- **Accuracy**: ~99.9%
- **F1-Score**: ~0.88

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
SPARK_HOME=/path/to/spark
JAVA_HOME=/path/to/java
FLASK_ENV=development
FLASK_DEBUG=True
```

### Spark Configuration

For optimal performance, adjust these settings in the training scripts:

```python
spark = SparkSession.builder \
    .appName("FraudDetection") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()
```

## 🚀 Deployment

### Docker Deployment

1. **Create Dockerfile**:
   ```dockerfile
   FROM python:3.9-slim
   
   RUN apt-get update && apt-get install -y openjdk-8-jdk
   ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
   
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   
   COPY . .
   
   EXPOSE 5000
   CMD ["python", "app/app.py"]
   ```

2. **Build and Run**:
   ```bash
   docker build -t fraud-detection .
   docker run -p 5000:5000 fraud-detection
   ```

### Cloud Deployment

- **AWS**: Deploy on EC2 with ELB for load balancing
- **GCP**: Use App Engine or Cloud Run
- **Azure**: Deploy on App Service

## 🧪 Testing

```bash
# Install test dependencies
pip install pytest pytest-flask

# Run tests
pytest tests/

# Run with coverage
pytest --cov=app tests/
```

## 📈 Monitoring

### Metrics to Track

- **Model Performance**: AUC, Precision, Recall
- **API Performance**: Response time, throughput
- **Business Metrics**: False positive rate, fraud catch rate

### Logging

The application includes comprehensive logging:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Commit your changes: `git commit -am 'Add some feature'`
5. Push to the branch: `git push origin feature-name`
6. Submit a pull request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Troubleshooting

### Common Issues

1. **Java Not Found**:
   - Install Java 8 JDK
   - Set JAVA_HOME environment variable

2. **Memory Issues**:
   - Reduce Spark executor memory
   - Use smaller dataset for testing

3. **Model Not Found**:
   - Ensure models are trained and saved in `models/` directory
   - Check file permissions

4. **Port Already in Use**:
   ```bash
   # Kill process using port 5000
   lsof -ti:5000 | xargs kill -9
   ```

### Getting Help

- 📧 Email: your-email@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/fraud-detection/issues)
- 📖 Documentation: [Wiki](https://github.com/yourusername/fraud-detection/wiki)

## 🏆 Acknowledgments

- Credit Card Fraud Detection Dataset from Kaggle
- Apache Spark MLlib documentation
- Flask documentation
- Bootstrap for UI components

## 🔮 Future Enhancements

- [ ] Real-time streaming fraud detection
- [ ] Advanced feature engineering
- [ ] Deep learning models (Neural Networks)
- [ ] Model explainability (SHAP, LIME)
- [ ] A/B testing framework
- [ ] Automated model retraining
- [ ] Multi-language support
- [ ] Mobile application
