# Smart Home Energy Consumption Prediction Dashboard (IT351)

An interactive IoT data science dashboard built with **Python**, **Streamlit**, and **Scikit-Learn** to monitor household electricity usage, analyze historical trends, and predict future energy consumption using Linear Regression.

## Live Demo
- **Interactive Dashboard:** [Open Streamlit App](https://it351-iot-project-9yifa9d2ltrgrjafjjnzkt.streamlit.app)

## Dashboard Features
1. **Dashboard Overview:** Key consumption metrics, historical time-series trends, and household filtering.
2. **Energy Analysis:** Custom date-range filtering, summary statistics, and consumption distribution analysis.
3. **Prediction Engine:** Real-time future energy consumption prediction based on current reading (kWh) and Household ID.
4. **Model Performance:** Evaluation metrics ($R^2 = 0.8511$, $\text{MSE} = 0.000840$) and actual vs. predicted comparison plots.

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
