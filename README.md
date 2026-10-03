Projectile Range Prediction Using Machine Learning

A machine learning study of projectile range prediction from initial velocity and launch angle using synthetic data generated from the analytical equations of ideal projectile motion.

Overview

This project explores how different regression algorithms learn the relationship between two physical input variables—initial velocity and launch angle—and projectile range.

Rather than manually specifying a prediction dataset, synthetic observations were generated from the classical projectile-motion model and then used to train and evaluate multiple machine learning algorithms.

The project serves as an introduction to applying machine learning within a physics-based problem while maintaining a reproducible scientific workflow.

Physical Model

For ideal projectile motion with equal launch and landing heights and negligible air resistance, the horizontal range is

[
R = \frac{v_0^2\sin(2\theta)}{g}
]

where:

* (R) = projectile range
* (v_0) = initial velocity
* (\theta) = launch angle
* (g) = gravitational acceleration

The machine learning models are therefore trained to approximate the mapping:

[
(v_0,\theta) \rightarrow R
]

Dataset

The dataset was generated computationally using randomly sampled:

* Initial velocity: 5–50 m/s
* Launch angle: 5–85°
* Number of observations: 5,000

A fixed random seed was used to make the experiment reproducible.

Machine Learning Models

Five regression approaches were evaluated:

1. Linear Regression
2. Polynomial Regression
3. Decision Tree Regression
4. Random Forest Regression
5. Gradient Boosting Regression

The models were trained using an 80/20 training-test split.

Evaluation

Model performance was evaluated using:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* Coefficient of Determination ((R^2))

Lower MAE, MSE and RMSE indicate smaller prediction errors, while an (R^2) closer to 1 indicates that the model explains a larger proportion of the variance in the test data.

Limitations

This experiment uses idealized synthetic data. It does not account for:

* Air resistance
* Wind
* Variations in gravitational acceleration
* Measurement uncertainty
* Projectile rotation or aerodynamic effects

Consequently, the results should not be interpreted as evidence of real-world predictive performance.

Future Work

Possible extensions include:

* Testing additional regression algorithms
* Cross-validation and systematic hyperparameter optimization
* Introducing realistic measurement noise
* Incorporating air resistance
* Comparing ML predictions with numerical and analytical solutions
* Extending the framework to more complex physical systems

Technologies

Python · NumPy · Pandas · Matplotlib · Scikit-learn

Author

Durodola Daniel