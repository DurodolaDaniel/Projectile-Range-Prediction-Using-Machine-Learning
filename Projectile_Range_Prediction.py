from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd

np.random.seed(42)
velocity=np.random.uniform(5,50,5000)
angle=np.random.uniform(5,85,5000)
Range=((velocity**2)*np.sin(2*(np.radians(angle))))/9.8
Dataset=pd.DataFrame({"velocity":velocity,"angle":angle,"Range":Range})
X=Dataset[["velocity","angle"]]
y=Dataset["Range"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
mae_1=mean_absolute_error(y_test,y_pred)
mse_1=mean_squared_error(y_test,y_pred)
rmse_1=mse_1**0.5
r2_1=r2_score(y_test,y_pred)

poly_reg=PolynomialFeatures(degree=2)
X_poly=poly_reg.fit_transform(X_train)
X_test_poly=poly_reg.transform(X_test)
model=LinearRegression()
model.fit(X_poly,y_train)
y_pred=model.predict(X_test_poly)
mae_2=mean_absolute_error(y_test,y_pred)
mse_2=mean_squared_error(y_test,y_pred)
rmse_2=mse_2**0.5
r2_2=r2_score(y_test,y_pred)

model=DecisionTreeRegressor(random_state=0)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
mae_3=mean_absolute_error(y_test,y_pred)
mse_3=mean_squared_error(y_test,y_pred)
rmse_3=mse_3**0.5
r2_3=r2_score(y_test,y_pred)

rf_model=RandomForestRegressor(n_estimators=500,random_state=42)
rf_model.fit(X_train,y_train)
rf_pred=rf_model.predict(X_test)
mae_4=mean_absolute_error(y_test,rf_pred)
mse_4=mean_squared_error(y_test,rf_pred)
rmse_4=mse_4**0.5
r2_4=r2_score(y_test,rf_pred)

model=GradientBoostingRegressor(learning_rate=0.1,n_estimators=500,max_depth=5,random_state=42)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
mae_5=mean_absolute_error(y_test,y_pred)
mse_5=mean_squared_error(y_test,y_pred)
rmse_5=mse_5**0.5
r2_5=r2_score(y_test,y_pred)

results=pd.DataFrame({"Model":["Linear Regression","Polynomial Regression","Decision Tree","Random Forest","Gradient Boosting"],
                   "MAE":[mae_1,mae_2,mae_3,mae_4,mae_5],"MSE":[mse_1,mse_2,mse_3,mse_4,mse_5],"RMSE":[rmse_1,rmse_2,rmse_3,
                   rmse_4,rmse_5],"R2":[r2_1,r2_2,r2_3,r2_4,r2_5]})
print(results)

x=["Linear Regression","Polynomial Regression","Decision Tree","Random Forest","Gradient Boosting"]
y=[r2_1,r2_2,r2_3,r2_4,r2_5]
plt.barh(x,y)
for i,value in enumerate(y):
    plt.text(value+0.01,i,f"{value:.4f}",va="center")
plt.title("R2 Comparison of Regression Models")
plt.xlabel("R2")
plt.xlim(0,1.1)
plt.show()

velocity_grid=np.linspace(5,50,100)
angle_grid=np.linspace(5,85,100)
V,A=np.meshgrid(velocity_grid,angle_grid)
grid=pd.DataFrame({
    "velocity":V.ravel(),
    "angle":A.ravel()
})
predicted_range=rf_model.predict(grid).reshape(V.shape)
plt.contourf(V,A,predicted_range,levels=20)
plt.colorbar(label="Predicted Range (m)")
plt.xlabel("Initial Velocity (m/s)")
plt.ylabel("Launch Angle (degrees)")
plt.title("Random Forest Prediction of Projectile Range")
plt.show()

plt.scatter(y_test,rf_pred)
plt.xlabel("Actual Range")
plt.ylabel("Predicted Range")
plt.title("Random Forest: Actual vs predicted Range")
plt.plot([min(y_test),max(y_test)],[min(y_test),max(y_test)])
plt.show()