Importing the dependencies
import numpy as np
import pandas as pd #pandas is useful to create data frames
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split #will help tp split training data and split data
from sklearn import svm
from sklearn.metrics import accuracy_score
Data Collection and Analysis
PIMA Dataset
#diabetes dataset in pandas tp create dataframe
data = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')

#printing the first 5 rows from the dataset
data.head()
Pregnancies	Glucose	BloodPressure	SkinThickness	Insulin	BMI	DiabetesPedigreeFunction	Age	Outcome
0	6	148	72	35	0	33.6	0.627	50	1
1	1	85	66	29	0	26.6	0.351	31	0
2	8	183	64	0	0	23.3	0.672	32	1
3	1	89	66	23	94	28.1	0.167	21	0
4	0	137	40	35	168	43.1	2.288	33	1
#showing the number of rows and columns in the dataset
data.shape 
(768, 9)
#getting the statistics of the dataset
data.describe()
Pregnancies	Glucose	BloodPressure	SkinThickness	Insulin	BMI	DiabetesPedigreeFunction	Age	Outcome
count	768.000000	768.000000	768.000000	768.000000	768.000000	768.000000	768.000000	768.000000	768.000000
mean	3.845052	120.894531	69.105469	20.536458	79.799479	31.992578	0.471876	33.240885	0.348958
std	3.369578	31.972618	19.355807	15.952218	115.244002	7.884160	0.331329	11.760232	0.476951
min	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.078000	21.000000	0.000000
25%	1.000000	99.000000	62.000000	0.000000	0.000000	27.300000	0.243750	24.000000	0.000000
50%	3.000000	117.000000	72.000000	23.000000	30.500000	32.000000	0.372500	29.000000	0.000000
75%	6.000000	140.250000	80.000000	32.000000	127.250000	36.600000	0.626250	41.000000	1.000000
max	17.000000	199.000000	122.000000	99.000000	846.000000	67.100000	2.420000	81.000000	1.000000
#helping to show the null value
data.isna().sum()
Pregnancies                 0
Glucose                     0
BloodPressure               0
SkinThickness               0
Insulin                     0
BMI                         0
DiabetesPedigreeFunction    0
Age                         0
Outcome                     0
dtype: int64
#finding the duplicate values from the dataset
data.duplicated().sum()
0
data['Outcome'].value_counts() #taking the outcome valueand checking how many outcomes are there for a particular label
Outcome
0    500
1    268
Name: count, dtype: int64
0 --> Non-Diabetic
1 --> Diabetic
#seperating the datadata.groupby('Outcome').mean()
#sperating thhe data and labels
X = data.drop(columns = 'Outcome',axis = 1)
Y = data['Outcome']
print(X)
     Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  \
0              6      148             72             35        0  33.6   
1              1       85             66             29        0  26.6   
2              8      183             64              0        0  23.3   
3              1       89             66             23       94  28.1   
4              0      137             40             35      168  43.1   
..           ...      ...            ...            ...      ...   ...   
763           10      101             76             48      180  32.9   
764            2      122             70             27        0  36.8   
765            5      121             72             23      112  26.2   
766            1      126             60              0        0  30.1   
767            1       93             70             31        0  30.4   

     DiabetesPedigreeFunction  Age  
0                       0.627   50  
1                       0.351   31  
2                       0.672   32  
3                       0.167   21  
4                       2.288   33  
..                        ...  ...  
763                     0.171   63  
764                     0.340   27  
765                     0.245   30  
766                     0.349   47  
767                     0.315   23  

[768 rows x 8 columns]
print(Y)
0      1
1      0
2      1
3      0
4      1
      ..
763    0
764    0
765    0
766    1
767    0
Name: Outcome, Length: 768, dtype: int64
Data Standardization
scaler = StandardScaler()
scaler.fit(X)
StandardScaler()
In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.
standardized_data = scaler.transform(X) #fitting inconstest data with standardscaler function, transforming all the data to a common range
print(standardized_data)
[[ 0.63994726  0.84832379  0.14964075 ...  0.20401277  0.46849198
   1.4259954 ]
 [-0.84488505 -1.12339636 -0.16054575 ... -0.68442195 -0.36506078
  -0.19067191]
 [ 1.23388019  1.94372388 -0.26394125 ... -1.10325546  0.60439732
  -0.10558415]
 ...
 [ 0.3429808   0.00330087  0.14964075 ... -0.73518964 -0.68519336
  -0.27575966]
 [-0.84488505  0.1597866  -0.47073225 ... -0.24020459 -0.37110101
   1.17073215]
 [-0.84488505 -0.8730192   0.04624525 ... -0.20212881 -0.47378505
  -0.87137393]]
X = standardized_data
Y = data['Outcome']
print(X)
print(Y)
[[ 0.63994726  0.84832379  0.14964075 ...  0.20401277  0.46849198
   1.4259954 ]
 [-0.84488505 -1.12339636 -0.16054575 ... -0.68442195 -0.36506078
  -0.19067191]
 [ 1.23388019  1.94372388 -0.26394125 ... -1.10325546  0.60439732
  -0.10558415]
 ...
 [ 0.3429808   0.00330087  0.14964075 ... -0.73518964 -0.68519336
  -0.27575966]
 [-0.84488505  0.1597866  -0.47073225 ... -0.24020459 -0.37110101
   1.17073215]
 [-0.84488505 -0.8730192   0.04624525 ... -0.20212881 -0.47378505
  -0.87137393]]
0      1
1      0
2      1
3      0
4      1
      ..
763    0
764    0
765    0
766    1
767    0
Name: Outcome, Length: 768, dtype: int64
Train Test Split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2) #will help to measure the unknownn data and its prediction accuracy
print(X.shape, X_train.shape, X_test.shape)
(768, 8) (614, 8) (154, 8)
Training the model
classifier = svm.SVC(kernel='linear') #svm = Support Vector Machine
#Training the Support Vector Machine classifier
classifier.fit(X_train, Y_train)
SVC(kernel='linear')
In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.
Model Evaluation
Accuracy Score

#accuracy score on the training data
X_train_predict = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_predict,Y_train)
print('Accuracy score of the training data: ',training_data_accuracy)
Accuracy score of the training data:  0.7866449511400652
#accuracy score on the test data

X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test)
print('Accuracy of the test data:', test_data_accuracy)
Accuracy of the test data: 0.7727272727272727
Making a Predictive System
input_data = (0,137,40,35,168,43.1,2.288,33)

#changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshaping the array according to the particular input
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

#standardizing the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

#checking the prediction with the given input
if(prediction[0] == 0):
    print('The person is not diabetic')
else:
    print('The person is diabetic')
[[-1.14185152  0.5040552  -1.50468724  0.90726993  0.76583594  1.4097456
   5.4849091  -0.0204964 ]]
[1]
The person is diabetic
