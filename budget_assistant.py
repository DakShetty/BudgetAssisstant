import os
import datetime
import hashlib

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from fpdf import FPDF
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Smart Budgeting Assistant", page_icon="💰", layout="wide")

THEME_CSS = """
<style>
body {
    background: linear-gradient(180deg, #f7f9fc 0%, #ffffff 45%);
    color: #1f2933;
    font-family: "Segoe UI", sans-serif;
}
.app-container {
    padding: 2rem 3rem;
    background-color: rgba(255, 255, 255, 0.88);
    border-radius: 18px;
    box-shadow: 0 25px 45px rgba(15, 23, 42, 0.10);
}
.auth-hero {
    padding: 3.5rem 4rem;
    text-align: center;
    background: radial-gradient(circle at top, #3b82f6 0%, #1d4ed8 60%, #111827 100%);
    color: #ffffff;
    border-radius: 24px;
    margin-bottom: 2.5rem;
}
.auth-hero h1 {font-size: 2.8rem; margin-bottom: 0.5rem;}
.auth-hero p {font-size: 1.1rem; opacity: 0.85;}
.stTabs [role="tablist"] button {
    padding: 0.8rem 1.2rem;
    border-radius: 999px;
    border: 1px solid #c7d2fe;
}
.metric-card {
    border-radius: 18px;
    padding: 1.25rem;
    background: #f8fafc;
    border: 1px solid #e0e7ff;
    box-shadow: inset 0 1px 0 rgba(99, 102, 241, 0.06);
}
.metric-card h3 {margin: 0; font-size: 1rem; color: #4c51bf;}
.metric-card p {margin: 0.2rem 0 0; font-size: 1.6rem; font-weight: 700;}
.alert-card {
    background: #fff7ed;
    border: 1px solid #fdba74;
    border-radius: 14px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
    font-weight: 500;
}
.success-card {
    background: #ecfdf5;
    border: 1px solid #6ee7b7;
    border-radius: 14px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
    font-weight: 500;
    color: #047857;
}
.sidebar .sidebar-content {
    background: #f8fafc;
}
</style>
"""

st.markdown(THEME_CSS, unsafe_allow_html=True)

APP_TITLE = "Smart Budgeting Assistant"
USER_DB_PATH = "users.csv"
DATA_FILENAME = "multi_month_data.csv"
DATA_COLUMNS = ["Month", "Category", "Amount", "Income", "Goal"]
CATEGORIES = ["Food", "Transport", "Rent", "Shopping", "Entertainment", "Other"]
CATEGORY_BASELINE = {
    "Food": 0.18,
    "Transport": 0.10,
    "Rent": 0.30,
    "Shopping": 0.12,
    "Entertainment": 0.10,
    "Other": 0.20,
}
MONTH_FORMATS = ["%b %Y", "%B %Y", "%Y-%m", "%m/%Y"]


def ensure_user_db(path: str = USER_DB_PATH) -> None:
    if not os.path.exists(path):
        pd.DataFrame(columns=["Username", "Password"]).to_csv(path, index=False)


def load_users(path: str = USER_DB_PATH) -> pd.DataFrame:
    ensure_user_db(path)
    return pd.read_csv(path)


def save_users(df: pd.DataFrame, path: str = USER_DB_PATH) -> None:
    df.to_csv(path, index=False)


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def verify_password(raw_password: str, hashed_password: str) -> bool:
    return hash_password(raw_password) == hashed_password


def authenticate_user(username: str, password: str) -> bool:
    users = load_users()
    record = users[users["Username"] == username]
    if record.empty:
        return False
    return verify_password(password, record.iloc[0]["Password"])


def register_user(username: str, password: str) -> tuple[bool, str]:
    if not username or not password:
        return False, "Username and password are required."

    users = load_users()
    if username in users["Username"].values:
        return False, "Username already exists. Please choose another."

    hashed = hash_password(password)
    users.loc[len(users)] = [username, hashed]
    save_users(users)
    return True, "Account created successfully! Please log in."


def get_user_storage(username: str) -> str:
    folder = f"data_{username}"
    os.makedirs(folder, exist_ok=True)
    return folder


def load_budget_data(folder: str) -> pd.DataFrame:
    data_path = os.path.join(folder, DATA_FILENAME)
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return pd.DataFrame(columns=DATA_COLUMNS)


def save_budget_data(df: pd.DataFrame, folder: str) -> None:
    data_path = os.path.join(folder, DATA_FILENAME)
    df.to_csv(data_path, index=False)


def month_sort_key(label: str) -> datetime.datetime:
    for fmt in MONTH_FORMATS:
        try:
            return datetime.datetime.strptime(label, fmt)
        except ValueError:
            continue
    return datetime.datetime.max


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def compute_predictions(data: pd.DataFrame, categories: list[str]) -> dict[str, float]:
    """
    Predict next month's spending for each category using linear regression.
    Requires at least 2 months of data for accurate predictions.
    With 1 month of data, uses that as baseline.
    """
    predictions = {}
    for cat in categories:
        cat_data = data[data["Category"] == cat]
        if len(cat_data) >= 2:
            # Use linear regression for trend-based prediction
            cat_data = cat_data.reset_index(drop=True)
            X = np.arange(len(cat_data)).reshape(-1, 1)
            y = cat_data["Amount"].values
            model = LinearRegression().fit(X, y)
            pred = max(model.predict([[len(cat_data)]])[0], 0)
            predictions[cat] = pred
        elif len(cat_data) == 1:
            # With only 1 month, use that as baseline prediction
            predictions[cat] = max(cat_data["Amount"].iloc[0], 0)
        else:
            # No data available
            predictions[cat] = np.nan
    return predictions


def compute_recommended_limits(
    data: pd.DataFrame, categories: list[str], income: float, goal: float
) -> tuple[dict[str, float], float]:
    if income <= 0:
        return {cat: 0 for cat in categories}, 0

    available_to_allocate = max(income - goal, income * 0.75)
    available_to_allocate = max(available_to_allocate, 0)

    history = data.groupby("Category")["Amount"].mean() if not data.empty else pd.Series(dtype=float)
    history_total = history.sum()

    weights = {}
    for cat in categories:
        historical_ratio = safe_divide(history.get(cat, 0), history_total)
        baseline_ratio = CATEGORY_BASELINE.get(cat, 1 / len(categories))
        weights[cat] = 0.6 * historical_ratio + 0.4 * baseline_ratio

    weight_sum = sum(weights.values()) or 1.0
    recommended_limits = {cat: available_to_allocate * weights[cat] / weight_sum for cat in categories}

    return recommended_limits, available_to_allocate


def generate_alerts(
    current_data: pd.DataFrame,
    predictions: dict[str, float],
    recommended_limits: dict[str, float],
    income: float,
    goal: float,
) -> list[str]:
    alerts = []
    total_expense = current_data["Amount"].sum()
    available_budget = income - goal if income > 0 else 0

    for cat, limit in recommended_limits.items():
        if limit <= 0:
            continue
        spent = current_data[current_data["Category"] == cat]["Amount"].sum()
        utilisation = safe_divide(spent, limit)
        predicted_spend = predictions.get(cat, np.nan)

        if utilisation >= 1:
            alerts.append(f"❗ {cat}: Already over the suggested limit by Rs.{spent - limit:.0f}.")
        elif utilisation >= 0.8:
            alerts.append(f"⚠️ {cat}: {utilisation*100:.0f}% of the limit used. Slow down to stay on track.")

        if not np.isnan(predicted_spend) and predicted_spend > limit:
            alerts.append(
                f"🤖 {cat}: Forecast suggests Rs.{predicted_spend:.0f}, above the recommended Rs.{limit:.0f}."
            )

    if available_budget > 0 and total_expense > available_budget:
        alerts.append(
            f"🚨 Overall spend exceeds disposable budget (income minus savings goal) by Rs.{total_expense - available_budget:.0f}."
        )

    return alerts


LIFE_EVENTS = {
    "New Job / Promotion": {
        "description": "Income bump expected. Increase savings while allowing modest lifestyle upgrades.",
        "income_multiplier": 1.15,
        "goal_uplift": 0.10,
        "category_shifts": {"Shopping": 0.02, "Entertainment": 0.02, "Other": -0.02},
    },
    "Moving to Costly City": {
        "description": "Housing and transport typically rise. Rebalance toward essentials.",
        "income_multiplier": 0.95,
        "goal_uplift": -0.05,
        "category_shifts": {"Rent": 0.08, "Transport": 0.03, "Entertainment": -0.03},
    },
    "Starting a Family": {
        "description": "Expect higher recurring expenses. Preserve your emergency savings buffer.",
        "income_multiplier": 0.90,
        "goal_uplift": 0.05,
        "category_shifts": {"Food": 0.06, "Other": 0.04, "Entertainment": -0.04},
    },
}


def simulate_life_event(
    event_name: str,
    recommended_limits: dict[str, float],
    income: float,
    goal: float,
) -> dict[str, float | dict[str, float]]:
    event = LIFE_EVENTS.get(event_name)
    if not event or income <= 0:
        return {"income": income, "goal": goal, "limits": recommended_limits}

    adjusted_income = max(income * event["income_multiplier"], 0)
    adjusted_goal = max(goal + income * event["goal_uplift"], 0)

    limits = recommended_limits.copy()
    for cat, shift in event.get("category_shifts", {}).items():
        if cat in limits:
            limits[cat] = max(limits[cat] + income * shift, 0)

    total_limits = sum(limits.values()) or 1
    spending_capacity = max(adjusted_income - adjusted_goal, adjusted_income * 0.75)
    spending_capacity = max(spending_capacity, 0)

    normalized_limits = {
        cat: (limits[cat] / total_limits) * spending_capacity if total_limits else 0 for cat in limits
    }

    return {"income": adjusted_income, "goal": adjusted_goal, "limits": normalized_limits}


def format_currency(value: float) -> str:
    return f"Rs.{value:,.0f}"


def sanitize_for_pdf(text: str) -> str:
    """
    Remove or replace Unicode characters (emojis, special chars) that can't be encoded in latin-1.
    FPDF uses latin-1 encoding by default, which only supports characters 0-255.
    """
    # Replace common emojis with text equivalents
    emoji_replacements = {
        "❗": "[!]",
        "⚠️": "[WARNING]",
        "🤖": "[AI]",
        "🚨": "[ALERT]",
    }
    result = text
    for emoji, replacement in emoji_replacements.items():
        result = result.replace(emoji, replacement)
    
    # Remove any remaining non-latin-1 characters
    try:
        result.encode("latin-1")
        return result
    except UnicodeEncodeError:
        # Fallback: remove any characters that can't be encoded
        return result.encode("latin-1", errors="ignore").decode("latin-1")


def render_authentication() -> None:
    st.markdown(
        """
        <div class="auth-hero">
            <h1>Smart Budgeting Assistant</h1>
            <p>Create realistic budgets, stay ahead of overspending, and adapt instantly to life changes.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    login_tab, signup_tab = st.tabs(["Login", "Sign Up"])

    with login_tab:
        st.subheader("Welcome back")
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            remember_me = st.checkbox("Keep me signed in", value=False, key="login_remember")
            login_submit = st.form_submit_button("Log In")

        if login_submit:
            if authenticate_user(username, password):
                st.session_state.authenticated = True
                st.session_state.user = username
                st.session_state.remember_me = remember_me
                st.success(f"Welcome back, {username}! 🎉")
                st.rerun()
            else:
                st.error("Invalid username or password.")

    with signup_tab:
        st.subheader("Create a new account")
        with st.form("signup_form", clear_on_submit=False):
            new_username = st.text_input("Choose a username", key="signup_username")
            new_password = st.text_input("Choose a password", type="password", key="signup_password")
            confirm_password = st.text_input("Confirm password", type="password", key="signup_confirm")
            signup_submit = st.form_submit_button("Create Account")

        if signup_submit:
            if new_password != confirm_password:
                st.error("Passwords do not match. Please try again.")
            else:
                success, message = register_user(new_username, new_password)
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)


def render_main_app(username: str) -> None:
    user_folder = get_user_storage(username)
    data = load_budget_data(user_folder)

    default_month = st.session_state.get("last_month_entry", datetime.datetime.now().strftime("%b %Y"))
    last_income = int(data.iloc[-1]["Income"]) if not data.empty else 0
    last_goal = int(data.iloc[-1]["Goal"]) if not data.empty else 0

    with st.sidebar:
        st.title("Control Center")
        st.markdown(f"👤 **Logged in as:** `{username}`")
        st.divider()

        st.markdown("### Add or update monthly data")

        category_inputs: dict[str, float] = {}
        with st.form("data_capture_form", clear_on_submit=False):
            month_entry = st.text_input("Month label (e.g. Nov 2025)", value=default_month)
            income_input = st.number_input("Monthly income (Rs.)", min_value=0, step=1000, value=last_income)
            goal_input = st.number_input("Target savings (Rs.)", min_value=0, step=1000, value=last_goal)

            for cat in CATEGORIES:
                category_inputs[cat] = st.number_input(
                    f"{cat} expenses (Rs.)", min_value=0, step=100, value=0, key=f"{cat}_input"
                )

            submitted = st.form_submit_button("Save month snapshot")

        if submitted:
            month_label = month_entry.strip()
            if not month_label:
                st.error("Please provide a month label before saving.")
            elif income_input <= 0:
                st.error("Please enter a valid monthly income (greater than 0) before saving.")
            else:
                new_rows = []
                for cat, amount in category_inputs.items():
                    new_rows.append(
                        {
                            "Month": month_label,
                            "Category": cat,
                            "Amount": amount,
                            "Income": income_input,
                            "Goal": goal_input,
                        }
                    )

                new_month_df = pd.DataFrame(new_rows)
                filtered_data = data[data["Month"] != month_label]
                updated_data = pd.concat([filtered_data, new_month_df], ignore_index=True)
                save_budget_data(updated_data, user_folder)
                st.session_state.last_month_entry = month_label
                st.success(f"Data for {month_label} saved successfully!")
                st.rerun()

        st.divider()
        if st.button("Log out", type="primary"):
            st.session_state.authenticated = False
            st.session_state.user = None
            st.session_state.remember_me = False
            st.success("You have been logged out.")
            st.rerun()

    st.markdown('<div class="app-container">', unsafe_allow_html=True)
    st.title(f"💰 {APP_TITLE}")
    st.caption("AI-powered budgeting that adapts to the way you live.")

    if data.empty:
        st.info("Start by adding your first month of data from the sidebar to unlock insights.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    month_options = list(dict.fromkeys(data["Month"].tolist()))
    selected_default = st.session_state.get("selected_month", month_options[-1])
    if selected_default not in month_options:
        selected_default = month_options[-1]
    selected_index = month_options.index(selected_default)
    selected_month = st.selectbox("Select month to review", month_options, index=selected_index)
    st.session_state.selected_month = selected_month

    current_month_data = data[data["Month"] == selected_month]
    current_income = current_month_data["Income"].max() if not current_month_data.empty else 0
    current_goal = current_month_data["Goal"].max() if not current_month_data.empty else 0

    predictions = compute_predictions(data, CATEGORIES)
    recommended_limits, disposable_pool = compute_recommended_limits(
        data, CATEGORIES, current_income, current_goal
    )
    alerts = (
        generate_alerts(current_month_data, predictions, recommended_limits, current_income, current_goal)
        if current_income > 0
        else []
    )

    tab_dashboard, tab_insights, tab_life_planner, tab_reports = st.tabs(
        ["📊 Dashboard", "💡 Smart Insights", "🧭 Life Planner", "📄 Reports"]
    )

    with tab_dashboard:
        st.subheader(f"Month snapshot · {selected_month}")
        total_expense = current_month_data["Amount"].sum()
        savings = current_income - total_expense

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-card"><h3>Income</h3>', unsafe_allow_html=True)
            st.markdown(f"<p>{format_currency(current_income)}</p></div>", unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card"><h3>Total Spend</h3>', unsafe_allow_html=True)
            st.markdown(f"<p>{format_currency(total_expense)}</p></div>", unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card"><h3>Net Savings</h3>', unsafe_allow_html=True)
            st.markdown(f"<p>{format_currency(savings)}</p></div>", unsafe_allow_html=True)

        if savings < 0:
            st.error("You're overspending this month. Let’s identify categories to trim.")
        elif current_goal > 0 and savings < current_goal:
            st.warning(
                f"You're Rs.{current_goal - savings:.0f} short of your savings goal. Consider dialing back discretionary spend."
            )
        else:
            st.success("Great job! You're on track with your plan.")

        chart_cols = st.columns(2)
        if not current_month_data.empty:
            category_summary = (
                current_month_data.groupby("Category")["Amount"].sum().reset_index().sort_values("Amount", ascending=False)
            )
            with chart_cols[0]:
                fig_bar = px.bar(
                    category_summary,
                    x="Category",
                    y="Amount",
                    text_auto=True,
                    title="Spending by category",
                    color="Category",
                    color_discrete_sequence=px.colors.qualitative.Safe,
                )
                fig_bar.update_layout(showlegend=False, yaxis_title="Amount (Rs.)")
                st.plotly_chart(fig_bar, use_container_width=True)

            with chart_cols[1]:
                fig_pie = px.pie(
                    category_summary,
                    names="Category",
                    values="Amount",
                    title="Share of total spend",
                    hole=0.45,
                    color_discrete_sequence=px.colors.qualitative.Safe,
                )
                st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("### Recommended envelope")
        recommended_df = pd.DataFrame(
            {
                "Category": list(recommended_limits.keys()),
                "Suggested limit (Rs.)": [round(val, 0) for val in recommended_limits.values()],
                "Current spend (Rs.)": [
                    current_month_data[current_month_data["Category"] == cat]["Amount"].sum()
                    for cat in recommended_limits
                ],
            }
        )
        recommended_df["Utilisation (%)"] = recommended_df.apply(
            lambda row: safe_divide(row["Current spend (Rs.)"], row["Suggested limit (Rs.)"]) * 100
            if row["Suggested limit (Rs.)"] > 0
            else 0,
            axis=1,
        ).round(1)
        st.dataframe(recommended_df, use_container_width=True)
        
        # Show forecast preview if available
        months_count = data["Month"].nunique() if not data.empty else 0
        if months_count >= 1:
            st.markdown("### 📈 Next month forecast")
            forecast_preview = []
            for cat in recommended_limits:
                pred = predictions.get(cat, np.nan)
                if np.isnan(pred):
                    forecast_preview.append("—")
                else:
                    forecast_preview.append(format_currency(pred))
            
            forecast_preview_df = pd.DataFrame({
                "Category": list(recommended_limits.keys()),
                "Forecast": forecast_preview
            })
            st.dataframe(forecast_preview_df, use_container_width=True)
            if months_count == 1:
                st.caption("💡 Forecasts use your current month as baseline. More accurate predictions appear after 2+ months of data.")

        st.markdown("### Historical view")
        history_table = data.copy()
        history_table["MonthOrder"] = history_table["Month"].apply(month_sort_key)
        history_table = history_table.sort_values(["MonthOrder", "Category"]).drop(columns=["MonthOrder"])
        st.dataframe(history_table.tail(50), use_container_width=True)

        trend = (
            data.groupby("Month", as_index=False)["Amount"].sum().assign(MonthOrder=lambda df: df["Month"].apply(month_sort_key))
        )
        trend = trend.sort_values("MonthOrder")
        fig_trend = px.line(
            trend,
            x="Month",
            y="Amount",
            markers=True,
            title="Total spending trend",
            line_shape="spline",
            color_discrete_sequence=["#2563eb"],
        )
        fig_trend.update_layout(yaxis_title="Amount (Rs.)")
        st.plotly_chart(fig_trend, use_container_width=True)

        st.markdown("### Active alerts")
        if alerts:
            for alert in alerts:
                st.markdown(f'<div class="alert-card">{alert}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="success-card">No alerts – spending is on plan. 🎯</div>', unsafe_allow_html=True)

    with tab_insights:
        st.subheader("AI-generated insights")
        total_all_time = data["Amount"].sum()
        monthly_totals = data.groupby("Month")["Amount"].sum()
        avg_monthly = monthly_totals.mean()
        highest_category = (
            data.groupby("Category")["Amount"].sum().idxmax() if not data.empty else "N/A"
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Total spent (all time)", format_currency(total_all_time))
        col2.metric("Average monthly spend", format_currency(avg_monthly))
        col3.metric("Top category", highest_category)

        disposable_budget = disposable_pool
        current_spend = current_month_data["Amount"].sum()
        savings_gap = current_goal - max(current_income - current_spend, 0)
        col4, col5, col6 = st.columns(3)
        col4.metric("Disposable budget", format_currency(disposable_budget))
        col5.metric("Current spend", format_currency(current_spend))
        col6.metric("Savings gap", format_currency(max(savings_gap, 0)) if savings_gap > 0 else "On track")

        st.markdown("### Forecast vs. plan")
        forecast_values = []
        for cat in recommended_limits:
            pred = predictions.get(cat, np.nan)
            if np.isnan(pred):
                forecast_values.append("Need data")
            else:
                forecast_values.append(f"Rs.{round(pred, 0):.0f}")
        
        forecast_df = pd.DataFrame(
            {
                "Category": list(recommended_limits.keys()),
                "Suggested limit (Rs.)": [round(val, 0) for val in recommended_limits.values()],
                "Forecast next month": forecast_values,
            }
        )
        st.dataframe(forecast_df, use_container_width=True)
        
        # Show info about forecast updates
        months_count = data["Month"].nunique() if not data.empty else 0
        if months_count < 2:
            st.info("💡 **Forecast updates:** Predictions will appear after you add at least 2 months of data. With 1 month, it uses that as a baseline.")
        elif months_count == 2:
            st.success("✅ Forecasts are now active! They update automatically each time you save new monthly data.")

        if data["Month"].nunique() >= 2:
            ordered_months = sorted(data["Month"].unique(), key=month_sort_key)
            latest_month, previous_month = ordered_months[-1], ordered_months[-2]
            latest_total = monthly_totals.loc[latest_month]
            previous_total = monthly_totals.loc[previous_month]
            delta = latest_total - previous_total
            if delta > 0:
                st.info(
                    f"Spending increased by Rs.{delta:.0f} compared to {previous_month}. Review discretionary categories."
                )
            elif delta < 0:
                st.success(
                    f"Spending decreased by Rs.{abs(delta):.0f} compared to {previous_month}. Excellent progress!"
                )

        st.markdown("### Personalized tip")
        if highest_category == "Food":
            st.warning("🍲 Food spend is leading. Try batch cooking or meal planning to cut costs.")
        elif highest_category == "Shopping":
            st.warning("🛍️ Shopping is dominant. Consider a no-spend week challenge.")
        elif highest_category == "Rent":
            st.info("🏠 Rent is largest. Keep it near or below 40% of take-home pay when possible.")
        elif highest_category == "Entertainment":
            st.warning("🎬 Entertainment is high. Explore low-cost or free alternatives for a few weeks.")
        else:
            st.success("👏 Spending looks balanced. Keep reinforcing the habits that work.")

    with tab_life_planner:
        st.subheader("Stress-test your budget")
        st.write("Model income shifts and buffer plans before you make big moves.")

        if current_income <= 0:
            st.info("Enter income for the selected month to unlock scenario planning.")
        else:
            event_choice = st.selectbox("Life event", ["-- choose --"] + list(LIFE_EVENTS.keys()))
            custom_income_shift = st.slider("Expected income change (%)", min_value=-50, max_value=100, value=0, step=5)
            emergency_buffer = st.number_input("Extra emergency buffer (Rs.)", min_value=0, step=500, value=0)

            if event_choice != "-- choose --":
                scenario = simulate_life_event(event_choice, recommended_limits, current_income, current_goal)
            else:
                scenario = {"income": current_income, "goal": current_goal, "limits": recommended_limits}

            adjusted_income = scenario["income"] * (1 + custom_income_shift / 100)
            adjusted_goal = scenario["goal"] + emergency_buffer

            total_limits = sum(scenario["limits"].values()) or 1
            spending_capacity = max(adjusted_income - adjusted_goal, adjusted_income * 0.75)
            spending_capacity = max(spending_capacity, 0)
            scaled_limits = {
                cat: (scenario["limits"][cat] / total_limits) * spending_capacity for cat in scenario["limits"]
            }

            col1, col2, col3 = st.columns(3)
            col1.metric("Projected income", format_currency(adjusted_income))
            col2.metric("Savings target", format_currency(adjusted_goal))
            col3.metric("Spending capacity", format_currency(spending_capacity))

            planner_df = pd.DataFrame(
                {
                    "Category": list(scaled_limits.keys()),
                    "Projected limit (Rs.)": [round(val, 0) for val in scaled_limits.values()],
                    "Current limit (Rs.)": [round(recommended_limits.get(cat, 0), 0) for cat in scaled_limits],
                }
            )
            st.dataframe(planner_df, use_container_width=True)

            if event_choice != "-- choose --":
                st.info(LIFE_EVENTS[event_choice]["description"])

    with tab_reports:
        st.subheader("Download a tailored PDF report")
        st.write("Includes current month metrics, recommendations, and AI alerts.")

        if current_month_data.empty:
            st.info("Add expenses for the selected month to generate a report.")
        else:
            if st.button("Generate PDF report"):
                try:
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", "B", 18)
                    pdf.cell(0, 12, txt="Smart Budgeting Report", ln=True, align="C")
                    pdf.set_font("Arial", size=12)
                    pdf.cell(
                        0,
                        8,
                        txt=f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
                        ln=True,
                        align="C",
                    )
                    pdf.ln(6)

                    pdf.set_font("Arial", "B", 14)
                    pdf.cell(0, 10, txt=sanitize_for_pdf(f"Month: {selected_month}"), ln=True)
                    pdf.set_font("Arial", size=11)
                    pdf.cell(0, 8, txt=sanitize_for_pdf(f"Income: {format_currency(current_income)}"), ln=True)
                    pdf.cell(0, 8, txt=sanitize_for_pdf(f"Total spend: {format_currency(total_expense)}"), ln=True)
                    pdf.cell(0, 8, txt=sanitize_for_pdf(f"Savings: {format_currency(savings)}"), ln=True)
                    pdf.cell(0, 8, txt=sanitize_for_pdf(f"Savings goal: {format_currency(current_goal)}"), ln=True)
                    pdf.ln(4)

                    pdf.set_font("Arial", "B", 13)
                    pdf.cell(0, 10, txt="Predicted next month expenses", ln=True)
                    pdf.set_font("Arial", size=11)
                    for cat, val in predictions.items():
                        if not np.isnan(val):
                            pdf.cell(0, 7, txt=sanitize_for_pdf(f"{cat}: {format_currency(val)}"), ln=True)
                    pdf.ln(4)

                    pdf.set_font("Arial", "B", 13)
                    pdf.cell(0, 10, txt="Recommended limits", ln=True)
                    pdf.set_font("Arial", size=11)
                    for cat, val in recommended_limits.items():
                        pdf.cell(0, 7, txt=sanitize_for_pdf(f"{cat}: {format_currency(val)}"), ln=True)
                    pdf.ln(4)

                    if alerts:
                        pdf.set_font("Arial", "B", 13)
                        pdf.cell(0, 10, txt="Active alerts", ln=True)
                        pdf.set_font("Arial", size=11)
                        for alert in alerts:
                            pdf.multi_cell(0, 7, txt=sanitize_for_pdf(alert))

                    report_path = os.path.join(
                        user_folder, f"budget_report_{selected_month.replace(' ', '_')}.pdf"
                    )
                    pdf.output(report_path)
                    with open(report_path, "rb") as pdf_file:
                        st.download_button(
                            "Download report",
                            pdf_file,
                            file_name=f"Budget_Report_{selected_month}.pdf",
                            mime="application/pdf",
                        )
                    st.success("Report generated successfully!")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Unable to generate report: {exc}")

    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.user = None
        st.session_state.remember_me = False

    if st.session_state.authenticated and st.session_state.user:
        render_main_app(st.session_state.user)
    else:
        render_authentication()


if __name__ == "__main__":
    main()

