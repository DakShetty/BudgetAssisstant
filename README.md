# 💰 Smart Budgeting Assistant

A comprehensive, AI-powered budgeting application built with Streamlit that helps you create realistic budgets, track spending, and adapt to life changes. Features intelligent expense predictions, personalized recommendations, and detailed financial reports.

## ✨ Features

### 📊 Dashboard
- **Monthly Overview**: Track income, expenses, and savings at a glance
- **Interactive Charts**: Visualize spending patterns with bar charts and pie charts
- **Category Breakdown**: Detailed analysis of spending across different categories
- **Historical Trends**: Track spending trends over time with interactive line charts

### 🤖 AI-Powered Insights
- **Expense Predictions**: Machine learning-based forecasts for next month's spending using linear regression
- **Smart Recommendations**: Personalized budget limits based on historical data and baseline ratios
- **Active Alerts**: Real-time notifications when spending exceeds recommended limits
- **Forecast Analysis**: Compare predicted expenses with recommended budgets

### 🧭 Life Planner
- **Scenario Planning**: Model how life events affect your budget
- **Life Event Simulations**: Pre-configured scenarios for:
  - New Job / Promotion
  - Moving to Costly City
  - Starting a Family
- **Custom Adjustments**: Adjust income changes and emergency buffers
- **Projected Budgets**: See how your budget adapts to different scenarios

### 📄 PDF Reports
- **Comprehensive Reports**: Generate detailed PDF reports for any month
- **Export Functionality**: Download reports with all metrics, predictions, and alerts
- **Professional Formatting**: Clean, organized report layout

### 🔐 User Management
- **Secure Authentication**: User registration and login system
- **Password Hashing**: SHA-256 password encryption
- **Per-User Data**: Isolated data storage for each user
- **Session Management**: Persistent login sessions

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/smart-budgeting-assistant.git
   cd smart-budgeting-assistant
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   Or install manually:
   ```bash
   pip install streamlit pandas numpy plotly scikit-learn fpdf2
   ```

3. **Run the application**
   ```bash
   streamlit run budget_assistant.py
   ```

4. **Access the app**
   - The app will automatically open in your default web browser
   - If not, navigate to `http://localhost:8501`

## 📖 Usage

### First Time Setup

1. **Create an Account**
   - Click on the "Sign Up" tab
   - Choose a username and password
   - Confirm your password
   - Click "Create Account"

2. **Add Your First Month**
   - Use the sidebar to enter your monthly data:
     - Month label (e.g., "Jan 2025")
     - Monthly income
     - Target savings goal
     - Expenses for each category (Food, Transport, Rent, Shopping, Entertainment, Other)
   - Click "Save month snapshot"

### Daily Usage

- **View Dashboard**: See your current month's financial overview
- **Check Insights**: Review AI-generated predictions and recommendations
- **Plan Ahead**: Use the Life Planner to simulate different scenarios
- **Generate Reports**: Download PDF reports for record-keeping

### Categories

The app tracks spending across six categories:
- 🍲 **Food**: Groceries, dining out, food delivery
- 🚗 **Transport**: Fuel, public transport, vehicle maintenance
- 🏠 **Rent**: Housing costs, utilities
- 🛍️ **Shopping**: Retail purchases, online shopping
- 🎬 **Entertainment**: Movies, subscriptions, hobbies
- 📦 **Other**: Miscellaneous expenses

## 🛠️ Technologies Used

- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Plotly**: Interactive data visualization
- **Scikit-learn**: Machine learning for expense predictions
- **FPDF**: PDF report generation
- **Python**: Core programming language

## 📁 Project Structure

```
smart-budgeting-assistant/
│
├── budget_assistant.py      # Main application file
├── users.csv                # User authentication database
├── data_[username]/         # User-specific data folders
│   ├── multi_month_data.csv # Monthly expense data
│   └── budget_report_*.pdf  # Generated PDF reports
└── README.md                # This file
```

## 🔧 Configuration

### Customizing Categories

Edit the `CATEGORIES` list in `budget_assistant.py`:

```python
CATEGORIES = ["Food", "Transport", "Rent", "Shopping", "Entertainment", "Other"]
```

### Adjusting Baseline Ratios

Modify `CATEGORY_BASELINE` to change default spending ratios:

```python
CATEGORY_BASELINE = {
    "Food": 0.18,
    "Transport": 0.10,
    "Rent": 0.30,
    # ... adjust as needed
}
```

## 📊 How It Works

### Expense Prediction

The app uses **Linear Regression** to predict next month's spending:
- Requires at least 2 months of data for accurate predictions
- With 1 month of data, uses that as a baseline
- Analyzes trends across all categories

### Budget Recommendations

Recommendations are calculated using:
- **60%** weight on historical spending patterns
- **40%** weight on baseline category ratios
- Available budget = Income - Savings Goal (minimum 75% of income)

### Alerts System

The app generates alerts when:
- Spending exceeds recommended limits
- Spending reaches 80% of limit (warning)
- Predicted expenses exceed recommendations
- Total spending exceeds disposable budget

## 🎯 Best Practices

1. **Consistent Data Entry**: Add monthly data regularly for better predictions
2. **Realistic Goals**: Set achievable savings goals
3. **Review Alerts**: Pay attention to spending warnings
4. **Use Life Planner**: Plan ahead for major life changes
5. **Export Reports**: Keep PDF reports for tax and financial planning

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Icons and emojis for visual enhancement
- Machine learning powered by [Scikit-learn](https://scikit-learn.org/)

## 📧 Support

If you encounter any issues or have questions, please open an issue on GitHub.

---

**Made with ❤️ for better financial planning**

