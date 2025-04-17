**Objective**

The primary goal to make this project is to predict whether a person is diabetes or non-diabetes based on the medical diagonostic dataset.The model is trained using the PIMA Indians Diabetes dataset from kaggle. The binary classification of the model:

                
                 0 --> diabetic
                 1 --> non-diabetic

                 
Diabetes is a chronic metabolic disorder characterized by high blood glucose levels, leading to potential damage to organs like the heart, blood vessels, eyes, and kidneys. It occurs when the body either doesn't produce enough insulin or can't effectively use the insulin it does produce. Insulin is a hormone that regulates blood glucose. Using this model, we can help professionals by: 
                 
                 
                 1. Making faster predictions
                 2. Providing a second opinion
                 3. Detecting patterns in the data that may not be obvious

                 
Aim of the project:

Using the patient's data (like glucose level, BMI, age, insulin, etc.) to detect the presence of diabetes and then applying the data preprocessing techniques like handling missing values, normalization, and splitting datasets. Training a machine learning model (SVM in this case) to learn from past data and generalize well on unseen patient records. At last, evaluating model performance using accuracy metrics to ensure its reliability for practical applications.


**Dataset**

**Source:**   PIMA Indians Diabetes Dataset

**Features:**

                  1. Pregnancies
                  2. Glucose
                  3. Blood Pressure
                  4. Skin Thickness
                  5. Insulin
                  6. BMI
                  7. Diabetes Pedigree Function
                  8. Age
                  9. Target: Outcome (0 = Non-Diabetic, 1 = Diabetic)

**Dataset insights:**

                  1. Checking for null values and duplicate data
                  2. Using StandardScaler to normalize feature data
                  3. Label (Outcome) distribution was analyzed

**Model details:**

**Algorithm Used:** Support Vector Machine (SVM) with a linear kernel

**Steps:**

                   1. Data was split into training and testing sets (80% train, 20% test)
                   2. Standardization of features using StandardScaler
                   3. Model has been trained using svm.SVC(kernel='linear')


**Results:**

**Training Accuracy:**

                   1. It has been calculated using the accuracy_score on training predictions
                   2. Test Accuracy: It has also evaluated using accuracy_score on test predictions


**Conclusion**

In this project, we successfully developed a machine learning model to predict whether a person is diabetic or not using the PIMA Indians Diabetes Dataset. By leveraging the Support Vector Machine (SVM) algorithm with a linear kernel, the model was able to learn from patterns in medical data and classify patients with reasonable accuracy.

Key steps included:

                   1. Cleaning and exploring the dataset using pandas
                   2. Scaling features with StandardScaler for better model performance
                   3. Splitting the data into training and testing sets
                   4. Training an SVM classifier, which is well-suited for binary classification problems
                   5. Evaluating the model's performance using accuracy metrics

The final results showed that the model performs well on both training and unseen test data, making it a promising tool for preliminary diabetes risk prediction. It highlights the potential of machine learning in healthcare diagnostics, and lays the groundwork for future improvements using more advanced techniques or larger datasets.
