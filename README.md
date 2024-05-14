# Predicting General Health Using BRFSS 2021

This repository contains the code and documentation for the project "Predicting General Health Using BRFSS 2021" for the CS7389G course.

## Project Overview

This project aims to predict the general health of individuals using the BRFSS 2021 dataset with the help of machine learning. Additionally, a Perceptual User Interface (PUI) is developed to manage the machine learning model through a web interface.

## Author

- Javad Mokhtari Koushyar (koushyar [at] txstate [dot] edu)

## Dataset

The dataset used in this project is the Behavioral Risk Factor Surveillance System (BRFSS) 2021 dataset, produced by the Centers for Disease Control and Prevention (CDC). It includes data collected from adults aged 18 and over from all 50 states, the District of Columbia, and three U.S. territories.

## Methods and Models

### Data Pre-processing

- **Conversion**: The dataset was converted from XPT to CSV format.
- **Handling Missing Data**: Missing data was replaced with a constant value.
- **Re-coding**: Certain columns were re-coded to handle non-useful values.
- **Balancing**: The SMOTE technique was used to balance the dataset.

### Machine-Learning Model

- **Model**: RandomForestClassifier from scikit-learn.
- **Feature Selection**: Pearson’s Correlation Algorithm was used to select relevant features.
- **Hyperparameter Tuning**: Tuning improved model accuracy significantly.
- **Cross-Validation**: Used to ensure the model's robustness and generalization.

### System Design

- **Database**: MongoDB is used to store the dataset and hyperparameters.
- **Back-End**: Flask framework for handling web services and interacting with the database.
- **Front-End**: React.js and Material-UI for building a user-friendly PUI.

## Results and Findings

### Evaluation Metrics

- **Precision**
- **Recall**
- **F1-Score**

### Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| 1     | 0.69      | 0.64   | 0.65     |
| 2     | 0.76      | 0.76   | 0.76     |
| 3     | 0.93      | 0.94   | 0.93     |
| 4     | 0.85      | 0.88   | 0.81     |
| 5     | 0.81      | 0.79   | 0.80     |

### Baseline Model

A Multinomial Logistic Regression model was used as the baseline. The RandomForestClassifier outperformed the baseline in all evaluation metrics.

## Discussion and Conclusion

### Advantages

- **Configurable Through Online PUI**: The model can be managed and configured easily via the web interface.
- **Outperforms the Baseline**: The RandomForestClassifier showed better performance compared to the baseline model.

### Limitations

- **Data Imbalance**: The model still struggles with class imbalance, affecting prediction accuracy.
- **Training Through PUI**: Training through the PUI is not feasible due to time constraints leading to HTTP request timeouts.

### Future Work

- **Neural Network Models**: Exploring neural network models for potentially better results.
- **GPU Compatibility**: Porting the source code to be GPU-compatible for improved training performance.
- **Work Offloading**: Enhancing the PUI to handle model training more efficiently.

## Getting Started

### Back-End Setup

1. Create a virtual environment:
    ```bash
    virtualenv venv
    source venv/bin/activate
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the back-end server:
    ```bash
    python app.py
    ```

### Front-End Setup

1. Install Node.js and NPM.
2. Navigate to the `frontend` directory:
    ```bash
    cd frontend
    ```

3. Install dependencies:
    ```bash
    npm install --force
    ```

4. Run the front-end application:
    ```bash
    npm start
    ```

## Citation

If you use this work, please cite it as follows:

```bibtex
@misc{mokhtarikoushyar2023health,
  title={Predicting General Health Using BRFSS 2021},
  author={Javad Mokhtari Koushyar},
  year={2023},
  howpublished={\url{https://github.com/your-repo-url}},
  note={CS7389G Final Project, Texas State University}
}
