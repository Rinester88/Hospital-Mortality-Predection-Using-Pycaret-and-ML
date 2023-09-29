# Hospital-Mortality-Predection-Using-Pycaret-and-ML
![download](https://github.com/Rinester88/Hospital-Mortality-Predection-Using-Pycaret-and-ML/assets/111410933/4df3542c-74a6-47de-b052-b48f32e9daf6)


The main aim of Hospital Mortality Prediction using PyCaret is to develop a predictive model that can accurately predict the likelihood of mortality (death) for patients admitted to a hospital. This predictive model is typically built using machine learning techniques and data collected from various sources within the hospital, such as patient demographics, medical history, lab results, vital signs, and other relevant clinical data.
The specific goals and benefits of Hospital Mortality Prediction using PyCaret may include:

### Early Intervention:

By accurately predicting which patients are at a higher risk of mortality, healthcare providers can identify those patients who may require more immediate or intensive medical attention. This can potentially lead to early interventions and improved patient outcomes.

### Resource Allocation:
Hospitals can optimize resource allocation by focusing more attention and resources on patients with a higher predicted mortality risk. This can include allocating intensive care unit (ICU) beds, specialized medical staff, or other critical resources more efficiently.

### Treatment Planning:
Predictive models can help healthcare professionals in tailoring treatment plans for individual patients. For high-risk patients, more aggressive treatment options or closer monitoring may be recommended.

### Quality Improvement:
Hospitals can use these models to assess and improve the quality of care provided. By analyzing factors contributing to mortality risk, hospitals can identify areas for improvement in patient care protocols and practices.

### Research and Clinical Trials:
Predictive models can also be valuable for medical research and clinical trials. Researchers can use these models to stratify patient groups based on mortality risk, helping in the design and analysis of clinical studies.

### Resource Efficiency:
Predictive modeling can help reduce healthcare costs by ensuring that resources are allocated efficiently, preventing unnecessary treatments or hospitalizations for patients with a low risk of mortality.

### Risk Communication:
Hospitals and healthcare providers can use mortality predictions to communicate more effectively with patients and their families. It allows for informed discussions about prognosis and end-of-life care planning.

In summary, Hospital Mortality Prediction using PyCaret aims to enhance patient care, resource allocation, and healthcare decision-making by leveraging machine learning and predictive analytics to estimate the likelihood of mortality for hospitalized patients. It is a valuable application of data science and predictive modeling in the healthcare industry, contributing to better patient outcomes and healthcare system efficiency.

# Motivation for using ML models and pycaret in Hospital Mortality Predection
Using machine learning (ML) models and PyCaret for Hospital Mortality Prediction is motivated by several compelling reasons:

Improved Patient Care: The primary motivation is to enhance patient care and outcomes. ML models can provide healthcare professionals with valuable insights into a patient's risk of mortality, allowing for early intervention and personalized treatment plans. This can ultimately lead to better patient outcomes and increased chances of survival.

Resource Optimization: Hospitals often face resource constraints, such as limited ICU beds and medical staff. ML models can help allocate these resources more effectively by prioritizing patients at higher risk of mortality. This ensures that critical care is provided where it is needed most.

Reduction in Healthcare Costs: Predictive modeling can lead to cost savings by preventing unnecessary hospitalizations or treatments for patients with a low risk of mortality. It optimizes resource usage, reducing healthcare expenses and improving the overall efficiency of healthcare delivery.

Clinical Decision Support: ML models can serve as valuable decision support tools for healthcare professionals. They can aid in clinical decision-making by providing data-driven insights into patient risk factors, allowing for more informed and evidence-based decisions.

Quality Improvement: By analyzing the factors contributing to mortality risk, hospitals can identify areas for quality improvement in patient care protocols and practices. This can lead to better overall healthcare quality and patient safety.

Research and Innovation: The use of ML in hospital mortality prediction contributes to medical research and innovation. It allows for the identification of novel risk factors and the development of more advanced predictive models, potentially leading to new insights and treatment approaches.

Continuous Monitoring: ML models can continuously monitor patient data, providing real-time updates on mortality risk. This proactive approach enables healthcare providers to respond promptly to changing patient conditions.

Ethical Considerations: The ethical motivation lies in ensuring that patients receive the best possible care. ML models can help healthcare providers uphold their ethical responsibility to prioritize patient well-being and safety.

Data-Driven Healthcare: The healthcare industry is increasingly moving towards data-driven decision-making. ML models align with this trend by harnessing the power of data to provide actionable insights that can save lives and improve patient care.

Global Health Challenges: In the face of global health challenges like pandemics, ML models for mortality prediction can be crucial in identifying high-risk individuals and allocating resources efficiently to mitigate the impact of the crisis.

In summary, the motivation for using ML models and PyCaret in Hospital Mortality Prediction is deeply rooted in the desire to enhance patient care, optimize healthcare resources, reduce costs, and uphold ethical standards in healthcare delivery. These models have the potential to save lives, improve healthcare quality, and drive innovation in the medical field.

# About pycaret and why are we using in Hospital Mortality Predection

PyCaret is an open-source Python library designed to streamline the end-to-end process of developing machine learning models. It simplifies many of the complex tasks associated with machine learning, making it easier for data scientists and analysts to build, evaluate, and deploy models. Here's an overview of PyCaret and why it's used in Hospital Mortality Prediction:

### Automated Machine Learning (AutoML)
PyCaret offers AutoML capabilities, allowing users to automate various steps in the machine learning workflow. This includes data preprocessing, feature selection, model training, hyperparameter tuning, and model evaluation. In the context of Hospital Mortality Prediction, this automation can save a significant amount of time and effort.

### Ease of Use 
PyCaret is designed to be user-friendly and intuitive. It provides a simple and consistent API that abstracts many of the complexities of machine learning, making it accessible to both beginners and experienced data scientists.

# Efficiency
Hospital Mortality Prediction often involves handling complex medical data. PyCaret's automation features speed up the process of experimenting with different models and techniques. This efficiency is crucial when dealing with time-sensitive healthcare decisions.

# Model Comparison
PyCaret enables users to easily compare the performance of multiple machine learning models, including traditional algorithms and ensemble methods. This is valuable for Hospital Mortality Prediction because it helps identify the model that best fits the specific data and problem.

# Hyperparameter Tuning
The library includes automated hyperparameter tuning, which optimizes the model's parameters to achieve the best possible performance. This is vital for creating accurate predictive models in healthcare scenarios where precision and recall are critical.

# Model Interpretability
Understanding why a model makes certain predictions is essential in healthcare. PyCaret provides tools for model interpretation and visualization, helping healthcare professionals and data scientists gain insights into the factors contributing to mortality predictions.

# Scalability
PyCaret supports both small and large datasets, making it suitable for analyzing healthcare data, which can vary widely in size and complexity.

# Deployment
Once a suitable model is developed, PyCaret facilitates model deployment, making it easier to integrate the predictive model into a hospital's information systems or electronic health records (EHR) for real-time predictions.

# Community and Support 
PyCaret has an active community and is well-documented. This means users can access resources, tutorials, and community support to address any issues or questions that arise during the model development process.

In Hospital Mortality Prediction, PyCaret simplifies the complexities of developing predictive models from medical data. By automating many of the steps and providing a range of machine learning tools, it empowers healthcare professionals and data scientists to create accurate and interpretable models for predicting patient outcomes, such as mortality. The library's focus on efficiency, model comparison, and deployment readiness makes it a valuable tool in healthcare analytics and decision support systems.

Community and Support: PyCaret has an active community and is well-documented. This means users can access resources, tutorials, and community support to address any issues or questions that arise during the model development process.

In Hospital Mortality Prediction, PyCaret simplifies the complexities of developing predictive models from medical data. By automating many of the steps and providing a range of machine learning tools, it empowers healthcare professionals and data scientists to create accurate and interpretable models for predicting patient outcomes, such as mortality. The library's focus on efficiency, model comparison, and deployment readiness makes it a valuable tool in healthcare analytics and decision support systems.

### Import Libraries
```
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
```
```

df=pd.read_csv(r'E:\Projects\Hospital Mortality Predection\data01.csv')
```
![image](https://github.com/Rinester88/Hospital-Mortality-Predection-Using-Pycaret-and-ML/assets/111410933/cc9ba32c-5856-4f13-9a4b-cae2f45bd002)

```
df.head()
```
![image](https://github.com/Rinester88/Hospital-Mortality-Predection-Using-Pycaret-and-ML/assets/111410933/602901c2-c19a-4831-842d-5f3be7cb4b25)
```
df.isnull().sum()
```
![image](https://github.com/Rinester88/Hospital-Mortality-Predection-Using-Pycaret-and-ML/assets/111410933/f4d03c25-c51d-42df-848d-379ee492a64a)
```
df.shape
```
```
df.info
```
![image](https://github.com/Rinester88/Hospital-Mortality-Predection-Using-Pycaret-and-ML/assets/111410933/c45a0b9d-5af2-4248-aebd-956c40264502)


# Handling NAN Values
### For Float Variables
```
from sklearn.impute import SimpleImputer 
si = SimpleImputer(missing_values= np.nan, strategy = 'mean')
```
```
float_col =df.select_dtypes(include='float64').columns
```
```
float_col
```
```
si.fit(df[float_col])
```
```
df[float_col] = si.transform(df[float_col])
```
```
x = df.drop(columns ='outcome')
y = df[['outcome']]
```
### For Dependent Variables
```
SI = SimpleImputer(missing_values = np.nan, strategy='most_frequent')
```
```
SI.fit_transform(y)
```
![image](https://github.com/Rinester88/Hospital-Mortality-Predection-Using-Pycaret-and-ML/assets/111410933/6c2e23d7-9ba8-474b-8ff8-ee0b48f32499)
```
y =pd.DataFrame(y, columns =['outcome'], dtype='int64')
```
```
import pandas as pd
import matplotlib.pyplot as plt

# Define df_final and populate it with data
df_final = pd.DataFrame({
    'outcome': ['Alive', 'Death', 'Alive', 'Alive', 'Death']
})

# Create a pie chart
fig, ax = plt.subplots(figsize=(9, 5), dpi=100)
patches, texts, autotexts = ax.pie(df_final['outcome'].value_counts(), autopct='%1.1f%%', shadow=True,
                                    startangle=90, explode=(0.1, 0), labels=['Alive', 'Death'])
plt.setp(autotexts, size=12, color='black', weight='bold')
autotexts[1].set_color('white')

# Add a title
plt.title('Outcome Distribution', fontsize=14)

# Display the chart
plt.show()
```
![image](https://github.com/Rinester88/Hospital-Mortality-Predection-Using-Pycaret-and-ML/assets/111410933/09756e3a-028e-4675-a534-9ed86e6aeec2)

```
import plotly.express as px

# Ensure the column name matches the one in your DataFrame
# If there's a space in the column name, use backticks (`) around the name
# Example: ` age`
fig = px.histogram(df, x="age", color="outcome", marginal="box", hover_data=df.columns)
fig.show()
```
```
df['gendera'].value_counts()
```
```
plt.figure(figsize=(10, 8))

# Corrected the column name to 'gendera'
plot = sns.countplot(df_final['gendera'], hue=df_final['outcome'])
plt.xlabel('Gender', fontsize=14, weight='bold')
plt.ylabel('Count', fontsize=14, weight='bold')
plt.xticks(np.arange(2), ['Male', 'Female'], rotation="vertical", weight='bold')

for i in plot.patches:
    plot.annotate(format(i.get_height()),
                  (i.get_x() + i.get_width() / 2, i.get_height()), 
                  ha='center', va='center',
                  size=10, xytext=(0, 8),
                  textcoords="offset points")

plt.show()
```
```
plt.figure(figsize=(12, 6))
    sns.histplot(df_final, x="age", hue="outcome", element="step", common_norm=False)
    plt.xlabel("Age", fontsize=14, weight="bold")
    plt.ylabel("Count", fontsize=14, weight="bold")
    plt.title("Distribution of Age by Outcome", fontsize=16, weight="bold")
    plt.legend(title="Outcome", labels=["Alive", "Death"])
    plt.show()
```
![image](https://github.com/Rinester88/Hospital-Mortality-Predection-Using-Pycaret-and-ML/assets/111410933/466957c4-1f2d-4ff4-9e36-6e0e78ef0ba2)
```
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_final, x="outcome", y="NT-proBNP")
plt.xlabel("Outcome", fontsize=14, weight="bold")
plt.ylabel("NT-proBNP Levels", fontsize=14, weight="bold")
plt.title("Distribution of NT-proBNP Levels by Outcome", fontsize=16, weight="bold")
plt.xticks([0, 1], ["Alive", "Death"], rotation=0, weight="bold")
plt.show()
```
```
col = ['group', 'gendera', 'hypertensive', 'atrialfibrillation', 'CHD with no MI', 'diabetes', 'deficiencyanemias', 'depression', 'Hyperlipemia', 'Renal failure', 'COPD', 'outcome']
```
```
# Define a custom colormap with unique colors
custom_cmap = sns.color_palette("Set3", as_cmap=True)

# Assuming 'corr' is your correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap=custom_cmap)

plt.show()
```
![image](https://github.com/Rinester88/Hospital-Mortality-Predection-Using-Pycaret-and-ML/assets/111410933/8f410628-f98f-4d70-a3ee-eeae169f9d33)
#### Distribution of Continuous variable 
```
plt.figure(figsize=(10,5))

df_final['Blood calcium'].plot(kind='kde')
```
![image](https://github.com/Rinester88/Hospital-Mortality-Predection-Using-Pycaret-and-ML/assets/111410933/086b1570-60d9-4bc3-89b7-c53810763b76)

```
df_final.head()
```
```
# Select continuous variables to plot
continuous_vars = ['age', 'NT-proBNP', 'Creatinine']

# Create subplots for each variable
plt.figure(figsize=(15, 5))
for i, var in enumerate(continuous_vars, 1):
    plt.subplot(1, 3, i)
    sns.histplot(df_final[var], kde=True, color='skyblue')
    plt.title(f'Distribution of {var}')
    plt.xlabel(var)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
```
![image](https://github.com/Rinester88/Hospital-Mortality-Predection-Using-Pycaret-and-ML/assets/111410933/79db04cd-9a91-4315-a3f2-9a344d478a57)
```
# Select continuous variables to plot
continuous_vars = ['age', 'NT-proBNP', 'Creatinine', 'RBC']

# Create a box plot for each variable
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_final[continuous_vars], palette='Set2')
plt.title('Box Plot of Continuous Variables')
plt.ylabel('Values')

plt.show()
```
```
# Select multiple continuous variables to plot
continuous_vars = ['age', 'NT-proBNP', 'Creatinine', 'RBC']

# Create a pair plot for selected variables
sns.pairplot(df_final[continuous_vars])
plt.suptitle('Pair Plot of Continuous Variables', y=1.02)
plt.show()
```
![image](https://github.com/Rinester88/Hospital-Mortality-Predection-Using-Pycaret-and-ML/assets/111410933/da3d6797-7db9-4b38-9cf9-20503301ce51)

## Data Pre-processing
#### Splitting the Data 
```
x = df_final.drop(columns='outcome')
y = df_final[['outcome']]
```
## Standardizing our data 
```
from sklearn.preprocessing import StandardScaler
```
```
scale = StandardScaler()

scaled =scale.fit_transform(x)
```
```
scaled
```
```
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=123)
```
```
x_train.drop(columns= 'ID', inplace =True)
x_test.drop(columns='ID' , inplace =True)
```
## Model Devlopment using ML 

#### We will using the XG Boost Classifier model
```
from xgboost import XGBClassifier, plot_tree, plot_importance
```
```
xgb = XGBClassifier(random_state=42)
```
```
xgb.fit(x_train, y_train)
```
```
pred =xgb.predict(x_test)
```
```
pred
```
```
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
```
```
import numpy as np

unique_y_test = np.unique(y_test)
unique_pred = np.unique(pred)

print("Unique values in y_test:", unique_y_test)
print("Unique values in pred:", unique_pred)
```
```
# Define a threshold (e.g., 0.5) to convert continuous values to binary
threshold = 0.5

# Apply the threshold to y_test
y_test_binary= (y_test >= threshold).astype(int)

# Now, both y_test_binary and pred have the same data type
```
```
cf= confusion_matrix(y_test_binary, pred)
```
```
print(classification_report(y_test_binary, pred))
```
## Plotting ROC and AccuracyCurve
```
from sklearn.metrics import plot_roc_curve, plot_precision_recall_curve
```
```
x_test
```
```
y_test
```
```
plot_roc_curve(xgb, x_test, y_test_binary)
plt.plot([0,1],[0,1], color = 'magenta', ls='-')
```
![image](https://github.com/Rinester88/Hospital-Mortality-Predection-Using-Pycaret-and-ML/assets/111410933/de5ac82f-f837-4352-918d-9e64cf0d4737)
# Using Auto ML 
#### Pycaret

PyCaret offers a high-level API for AutoML, allowing you to perform tasks such as data preprocessing, feature selection, model training, hyperparameter tuning, and model evaluation with minimal code. It automates many of the repetitive tasks in machine learning. 

![download](https://github.com/Rinester88/Hospital-Mortality-Predection-Using-Pycaret-and-ML/assets/111410933/6b706d03-84fa-4a08-929f-5114d58ae96b)

```
!pip install pycaret
```
```
df=pd.read_csv(r'E:\Projects\Hospital Mortality Predection\data01.csv')
```
```
from pycaret.classification import *
```
```
model = setup(data =df, target = 'outcome')
```
```
compare_models()
```
```
lda = create_model('lda')
```
![image](https://github.com/Rinester88/Hospital-Mortality-Predection-Using-Pycaret-and-ML/assets/111410933/5db59c92-76b2-440e-bfe4-8313b15f5971)

```
pred =predict_model(lda,data=x_test)
```
```
pred
```
![image](https://github.com/Rinester88/Hospital-Mortality-Predection-Using-Pycaret-and-ML/assets/111410933/ee9fadd9-d564-4a58-8cca-4da8a9219c5d)











