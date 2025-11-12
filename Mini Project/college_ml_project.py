# college_ml_project_final3.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -------------------------------
# STEP 1: Load datasets
# -------------------------------
try:
    df_raw = pd.read_csv("kaggle_sw_raw.csv")
    df_pivot = pd.read_csv("kaggle_pivot_min_descending.csv")
    print("✅ Both datasets loaded successfully!")
except FileNotFoundError as e:
    print("❌ File not found! Please check CSV filenames.")
    print(e)
    exit()

print("\nRaw dataset shape:", df_raw.shape)
print("Pivot dataset shape:", df_pivot.shape)

# -------------------------------
# STEP 2 & 3: Clean and convert dataset
df = df_raw.copy()

# Create target column: rank <= 5000 -> admitted
df['admitted'] = df['rank'].apply(lambda x: 1 if x <= 5000 else 0)

# Drop unnecessary columns
df = df.drop(['enrollment_no', 'branch_code'], axis=1, errors='ignore')

# Replace invalid symbols in object columns with 'Unknown'
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype(str).str.strip()
    df[col] = df[col].replace(['^', '~', '*', '&', '@', 'nan', 'NaN'], 'Unknown')

# Drop rows with any missing values
df = df.dropna()

# Convert all object/string columns to numeric using get_dummies
categorical_cols = df.select_dtypes(include='object').columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Now all features are numeric
X = df.drop('admitted', axis=1)
y = df['admitted']

print("Rows after cleaning:", df.shape[0])
print("All feature columns are numeric:", all([pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes]))

# -------------------------------
# STEP 4: Split dataset
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# -------------------------------
# STEP 5: Train Decision Tree
# -------------------------------
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
print("✅ Model trained successfully!")

# -------------------------------
# STEP 6: Evaluate model
# -------------------------------
y_pred = model.predict(X_test)
print("\n🎯 Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------------------
# STEP 7: Analyze pivot dataset
# -------------------------------
print("\n📊 Top 5 Colleges by Average Mean Score:")
top_mean = df_pivot.groupby('college_name')['mean'].mean().sort_values(ascending=False).head(5)
print(top_mean)

print("\n🏫 Colleges with Highest Score Difference (max-min):")
top_diff = df_pivot[['college_name','max-min']].sort_values(by='max-min', ascending=False).head(5)
print(top_diff)

# -------------------------------
# STEP 8: Predict new student admission
# -------------------------------
def predict_student():
    print("\n🎓 Enter new student details to check admission:")

    try:
        rank = int(input("Rank: "))
        percentile = float(input("Percentile (0-100): "))
        branch = input("Branch (Computer/Mechanical/Electrical/Civil/etc): ")
        gender = input("Gender (M/F): ").upper()
        category = input("Category (OPEN/SC/ST/OBC/EWS): ").upper()
        seat_type = input("Seat Type (CAP/AI/DEF/JK/etc): ")
        score_type = input("Score Type (CET/JEE): ")
        college_name = input("College Name: ")

        # Build input dictionary
        data = {
            'rank': rank,
            'percentile': percentile,
            # Gender
            'gender_M': 1 if gender == 'M' else 0,
            # Category
            'category_OBC': 1 if category == 'OBC' else 0,
            'category_SC': 1 if category == 'SC' else 0,
            'category_ST': 1 if category == 'ST' else 0,
            # Seat type
            'seat_type_CAP': 1 if seat_type == 'CAP' else 0,
            # Score type
            'score_type_JEE': 1 if score_type == 'JEE' else 0
        }

        # Dynamically add college columns
        college_columns = [col for col in X.columns if col.startswith('college_name_')]
        for col in college_columns:
            data[col] = 1 if col == f'college_name_{college_name}' else 0

        input_df = pd.DataFrame([data])

        input_df = input_df.reindex(columns=X.columns, fill_value=0)



        # Predict
        pred = model.predict(input_df)[0]
        if pred == 1:
            print("\n✅ Result: Student likely gets admission!")
        else:
            print("\n❌ Result: Student may NOT get admission.")

    except Exception as e:
        print("⚠️ Error in input:", e)

# Run interactive prediction
predict_student()
