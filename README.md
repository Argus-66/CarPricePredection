# 🚗 Car Price Prediction System

A comprehensive machine learning solution for predicting used car prices with an interactive Streamlit web application.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Dataset Information](#dataset-information)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

The used car market is rapidly expanding due to increasing vehicle ownership and the demand for affordable mobility solutions. Determining a fair resale value is challenging since it depends on multiple factors such as brand, model, year, mileage, fuel type, transmission, and market demand.

This project develops a **Machine Learning-powered system** that predicts used car prices with high accuracy using real-world data from CarDekho. The solution includes comprehensive data analysis, multiple ML models, and an interactive web dashboard.

### 🎯 Problem Statement
- Traditional pricing methods rely on dealer judgment or manual inspection
- Leads to inconsistencies and lack of transparency
- Need for data-driven, objective pricing mechanism

### 🎯 Solution
- **Gradient Boosting Regressor** model achieving **R² of 0.87**
- Interactive **Streamlit web application** for real-time predictions
- Comprehensive data analysis and visualization tools
- User-friendly interface for car comparison and filtering

## ✨ Features

### 🔍 **Data Filtering & Exploration**
- Filter cars by brand, model, fuel type, transmission
- Year range filtering with dynamic model updates
- Interactive data tables with sorting capabilities

### 📊 **Data Analysis Dashboard**
- Brand-wise distribution analysis
- Fuel type and transmission statistics
- Price trend analysis across different car attributes
- Interactive Plotly visualizations

### 💰 **Price Prediction Engine**
- Real-time price prediction for any car configuration
- Input validation with launch year checking
- Confidence intervals and prediction accuracy metrics
- Historical price comparison

### 📈 **Car Comparison Tool**
- Side-by-side comparison of multiple car models
- Interactive scatter plots and bar charts
- Price vs. features analysis
- Performance metrics comparison

### 🎨 **Modern UI/UX**
- Glassmorphism design with background images
- Responsive layout for all screen sizes
- Dark theme with gradient text effects
- Intuitive navigation with emoji icons

## 🛠️ Technology Stack

### **Backend & ML**
- **Python 3.8+** - Core programming language
- **Pandas & NumPy** - Data manipulation and analysis
- **Scikit-learn** - Machine learning algorithms
- **Joblib** - Model serialization

### **Frontend & Visualization**
- **Streamlit** - Web application framework
- **Plotly** - Interactive data visualizations
- **Matplotlib & Seaborn** - Statistical plotting
- **HTML/CSS** - Custom styling

### **Data Processing**
- **Jupyter Notebook** - Data exploration and preprocessing
- **Label Encoding** - Categorical variable handling
- **Feature Engineering** - Car age, price per km calculations

## 📁 Project Structure

```
CarPricePrediction/
├── 📊 Data Files
│   ├── car_dataset.csv              # Original dataset
│   ├── Preprocessed_Car_Dheko.csv   # Cleaned dataset
│   └── inferred_launch_years.csv    # Car launch year data
│
├── 🤖 Machine Learning Models
│   ├── GradientBoost_model.pkl      # Trained Gradient Boosting model
│   ├── DecisionTreeRegressor_model.pkl # Alternative model
│   └── label_encoders.pkl           # Categorical encoders
│
├── 🖥️ Streamlit Application
│   ├── Main.py                      # Main application entry point
│   ├── Home.py                      # Landing page with hero section
│   ├── Filtering.py                 # Data filtering interface
│   ├── Analysis.py                  # Data analysis dashboard
│   ├── Prediction.py                # Price prediction tool
│   └── Comparison.py                # Car comparison features
│
├── 📓 Data Science Notebooks
│   └── Data_Cleaning_Preprocessing.ipynb # Complete data pipeline
│
├── 🎨 Assets & Styling
│   ├── assets/263800.jpg            # Background image
│   └── style.css                    # Custom CSS styles
│
├── ⚙️ Configuration
│   ├── requirements.txt             # Python dependencies
│   ├── run_app.bat                  # Windows launch script
│   └── .gitignore                   # Git ignore rules
│
└── 📚 Documentation
    └── README.md                    # Project documentation
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (for cloning)

### Step 1: Clone the Repository
```bash
git clone https://github.com/Argus-66/CarPricePredection.git
cd CarPricePredection
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python -m venv myenv

# Activate virtual environment
# On Windows:
myenv\Scripts\activate
# On macOS/Linux:
source myenv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
# Check if all packages are installed
pip list
```

## 🎮 Usage

### Running the Application

1. **Start the Streamlit App:**
```bash
streamlit run Main.py
```

2. **Access the Application:**
   - Open your browser and go to `http://localhost:8501`
   - The application will automatically open in your default browser

### Using the Features

#### 🏠 **Home Page**
- Welcome screen with project overview
- Quick navigation to all features
- Modern glassmorphism design

#### 🔍 **Data Filtering**
- Select brands and models from dropdown menus
- Apply fuel type and transmission filters
- Set year ranges for car manufacturing
- View filtered results in interactive tables

#### 📊 **Data Analysis**
- Explore brand distribution charts
- Analyze fuel type preferences
- Study price trends across different categories
- Interactive visualizations with zoom and pan

#### 💰 **Price Prediction**
- Select car brand and model
- Input manufacturing year
- Choose fuel type, transmission, and other features
- Get instant price prediction with confidence metrics

#### 📈 **Car Comparison**
- Compare multiple car models side-by-side
- Interactive scatter plots for price vs. features
- Filter comparisons by various attributes
- Export comparison results

## 📈 Model Performance

### **Gradient Boosting Regressor**
- **R² Score:** 0.87 (87% variance explained)
- **Mean Absolute Error:** Optimized for real-world accuracy
- **Cross-Validation:** 5-fold validation implemented
- **Feature Importance:** Top features identified and utilized

### **Model Features Used:**
- **Vehicle Age** - Calculated from manufacturing year
- **Brand & Model** - Label encoded categorical variables
- **Engine Specifications** - Engine capacity and max power
- **Fuel Type** - Petrol, Diesel, CNG, Electric
- **Transmission** - Manual vs. Automatic
- **Mileage & Distance** - Fuel efficiency and kilometers driven
- **Seating Capacity** - Number of seats

### **Model Training Process:**
1. **Data Preprocessing** - Cleaning, outlier removal, missing value handling
2. **Feature Engineering** - Car age calculation, categorical encoding
3. **Model Selection** - Comparison of multiple algorithms
4. **Hyperparameter Tuning** - Grid search optimization
5. **Validation** - Cross-validation and holdout testing

## 📊 Dataset Information

### **Source:** CarDekho Platform
### **Size:** ~4,000+ used car records
### **Features:** 15+ attributes per car

### **Key Attributes:**
| Feature | Type | Description |
|---------|------|-------------|
| Brand | Categorical | Car manufacturer (Maruti, Honda, Toyota, etc.) |
| Model | Categorical | Specific car model |
| Year | Numerical | Manufacturing year |
| Selling_Price | Numerical | Target variable (₹ Lakhs) |
| KM_Driven | Numerical | Odometer reading |
| Fuel_Type | Categorical | Petrol/Diesel/CNG/Electric |
| Transmission | Categorical | Manual/Automatic |
| Engine | Numerical | Engine capacity (CC) |
| Max_Power | Numerical | Maximum power (bhp) |
| Mileage | Numerical | Fuel efficiency (kmpl) |
| Seats | Numerical | Seating capacity |

### **Data Quality:**
- **Missing Values:** Handled through imputation and removal
- **Outliers:** Detected and treated using IQR method
- **Categorical Encoding:** Label encoding for ML compatibility
- **Feature Scaling:** Normalized for optimal model performance

## 🖼️ Screenshots

### Home Dashboard
![Home Dashboard](https://via.placeholder.com/800x400?text=Home+Dashboard+Screenshot)

### Price Prediction Interface
![Price Prediction](https://via.placeholder.com/800x400?text=Price+Prediction+Interface)

### Data Analysis Charts
![Data Analysis](https://via.placeholder.com/800x400?text=Data+Analysis+Charts)

### Car Comparison Tool
![Car Comparison](https://via.placeholder.com/800x400?text=Car+Comparison+Tool)

## 🔮 Future Enhancements

### **Technical Improvements**
- [ ] Deep Learning models (Neural Networks, LSTM)
- [ ] Real-time data integration APIs
- [ ] Location-based pricing adjustments
- [ ] Image-based car condition assessment

### **Feature Additions**
- [ ] User authentication and saved searches
- [ ] Price alerts and notifications
- [ ] Market trend predictions
- [ ] Mobile application development

### **Data Enhancements**
- [ ] Larger dataset with more recent records
- [ ] Additional features (accident history, service records)
- [ ] Real-time market data integration
- [ ] Regional pricing variations

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### **Getting Started**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### **Contribution Guidelines**
- Follow PEP 8 Python style guidelines
- Add comments and docstrings to your code
- Include unit tests for new features
- Update documentation as needed

### **Areas for Contribution**
- Model improvement and optimization
- UI/UX enhancements
- New feature development
- Bug fixes and performance optimization
- Documentation improvements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Argus-66**
- GitHub: [@Argus-66](https://github.com/Argus-66)
- Project: [CarPricePredection](https://github.com/Argus-66/CarPricePredection)

## 🙏 Acknowledgments

- **CarDekho** for providing the dataset
- **Streamlit** community for excellent documentation
- **Scikit-learn** developers for robust ML tools
- **Plotly** team for interactive visualization capabilities

## 📞 Contact & Support

- **Issues:** [GitHub Issues](https://github.com/Argus-66/CarPricePredection/issues)
- **Discussions:** [GitHub Discussions](https://github.com/Argus-66/CarPricePredection/discussions)

---

<div align="center">

### ⭐ Star this repository if you found it helpful!

**Made with ❤️ using Python, Streamlit, and Machine Learning**

</div>
# CarPricePredection
# CarPricePredection
