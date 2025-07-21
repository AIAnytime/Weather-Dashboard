# Weather Dashboard Project

An interactive weather data visualization dashboard with EDA (Exploratory Data Analysis) capabilities built using Streamlit.

## Features

- Interactive filters for time periods, weather conditions, and more
- Key weather metrics and KPIs
- Temperature trend visualizations
- Rainfall analysis
- Humidity variation charts
- Wind patterns visualization
- Advanced EDA tools including correlation analysis and outlier detection

## Prerequisites

Before you begin, ensure you have Python installed on your computer. This project requires Python 3.11 or newer.

### Check if Python is installed

1. Open a terminal (Mac/Linux) or Command Prompt (Windows)
2. Type the following command and press Enter:

```
python --version
```

- If Python is installed, you'll see a message like `Python 3.12.4`
- If you see `Python 2.x.x`, try using `python3 --version` instead
- If you get an error, you need to install Python from [python.org](https://www.python.org/downloads/)

## Setup Instructions

### Step 1: Download the Project

- Download this project to your computer
- Extract the files if it's in a zip format

### Step 2: Open Terminal/Command Prompt

#### On Windows:
- Press `Win + R`, type `cmd` and press Enter
- Navigate to your project folder using the `cd` command
  ```
  cd path\to\weather-dashboard
  ```

#### On Mac/Linux:
- Open Terminal
- Navigate to your project folder using the `cd` command
  ```
  cd path/to/weather-dashboard
  ```

### Step 3: Create a Virtual Environment (Optional but Recommended)

#### On Windows:
```
python -m venv .venv
.venv\Scripts\activate
```

#### On Mac/Linux:
```
python -m venv .venv
source .venv/bin/activate
```

### Step 4: Install Required Packages

```
pip install -r requirements.txt
```

## Running the Dashboard

### Option 1: Main Dashboard

To run the full interactive dashboard with all features:

```
streamlit run app.py
```

### Option 2: Simple KPI Dashboard (via Jupyter Notebook)

1. Install Jupyter if not already installed:
   ```
   pip install jupyter
   ```

2. Start Jupyter Notebook:
   ```
   jupyter notebook
   ```

3. Open `weather_kpi_notebook.ipynb` in the Jupyter interface
4. Follow the instructions in the notebook to run the simplified KPI dashboard

## Viewing the Dashboard

After running the command, Streamlit will automatically open your default web browser to show the dashboard. If it doesn't open automatically, you can access it at:

```
http://localhost:8501
```

## Stopping the Dashboard

To stop the dashboard, press `Ctrl+C` in the terminal or command prompt where it's running.

## Troubleshooting

- **Error: Port is already in use**: Try running with a different port
  ```
  streamlit run app.py --server.port 8502
  ```

- **Package installation issues**: Make sure you're using a recent version of pip
  ```
  pip install --upgrade pip
  ```
  Then try installing the requirements again

- **"Module not found" errors**: Make sure you've installed all requirements
  ```
  pip install -r requirements.txt
  ```
