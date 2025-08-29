# ML-Dashboard
# 📊 Interactive Machine Learning Dashboard
This project is an interactive web application that allows users to select different datasets, choose a machine learning classification algorithm, and tune its hyperparameters in real-time to see how it affects model performance. The dashboard visualizes the dataset using PCA and displays the model's accuracy.

This tool is perfect for learning, demonstrating, and experimenting with some of the most common classification models in machine learning.

✨ Features
Interactive UI: Built with Streamlit for a clean and responsive user experience.

Multiple Datasets: Choose from classic ML datasets like Iris, Breast Cancer, and Wine.

Classifier Selection: Experiment with different algorithms including K-Nearest Neighbors (KNN), Support Vector Machine (SVM), and Random Forest.

Real-Time Hyperparameter Tuning: Use sliders and input boxes in the sidebar to adjust model parameters on the fly.

Performance Metrics: Instantly view the accuracy of the trained classifier on the test set.

Data Visualization: See a 2D PCA plot of the dataset to understand its structure.

🛠️ Tech Stack
Framework: Streamlit

Machine Learning: Scikit-learn

Data Manipulation: NumPy & Pandas

Plotting: Matplotlib

⚙️ Setup and Installation
Follow these steps to run the project on your local machine.

1. Clone the repository:

git clone [https://github.com/desaikesha13/ML-Dashboard.git](https://github.com/desaikesha13/ML-Dashboard.git)
cd ML-Dashboard

2. Create a virtual environment (recommended):
This keeps the project's dependencies isolated from your system's Python installation.

# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

3. Install the required dependencies:
The requirements.txt file contains all the necessary Python packages.

pip install -r requirements.txt

4. Run the Streamlit application:
This command will start the local server and open the application in your web browser.

streamlit run app.py

You should now be able to see and interact with the dashboard at http://localhost:8501.

▶️ How to Use
Once the application is running, a sidebar will appear on the left.

Use the "Select Dataset" dropdown to choose the dataset you want to work with.

Use the "Select Classifier" dropdown to pick a machine learning algorithm.

Based on the selected classifier, a set of hyperparameters will appear. Use the sliders and input boxes to adjust them.

The dashboard will automatically retrain the model with the new parameters and update the accuracy score and the PCA plot on the main screen.

Feel free to fork this repository, make changes, and open a pull request!
