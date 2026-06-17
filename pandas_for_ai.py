"""
Pandas for AI Engineering
Core pandas operations essential for data analysis and ML pipelines.
"""

import pandas as pd
import numpy as np

print("Pandas version:", pd.__version__)
print("NumPy version:", np.__version__)


# 1. SERIES - one column of labeled data
marks = pd.Series([85, 70, 90, 60, 75],
                  index=["avu", "chinnu", "minnu", "joy", "oggy"])
print("\n--- Series ---")
print(marks)
print("Marks datatype:", marks.dtype)
print("Shape:", marks.shape)
print("Index:", marks.index.tolist())
print("Avu's marks:", marks["avu"])
print("Marks above 75:\n", marks[marks > 75])


# 2. DATAFRAME - full table with multiple columns
data = {
    "Name": ["Avu", "Chinnu", "Minnu", "Joy", "Oggy"],
    "Age": [21, 22, 20, 23, 21],
    "Marks": [85, 70, 90, 60, 75],
    "City": ["Kannur", "Kochi", "Thrissur", "Kozhikode", "Trivandrum"],
    "Grade": ["A", "B", "A", "C", "B"]
}
df = pd.DataFrame(data)
print("\n--- DataFrame ---")
print(df)
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Data types:\n", df.dtypes)


# 3. EXPLORING DATA
print("\n--- Exploring Data ---")
print("First 3 rows:\n", df.head(3))
print("Last 2 rows:\n", df.tail(2))
print("Dataset info:")
df.info()
print("Statistical summary:\n", df.describe())
print("Missing values:\n", df.isnull().sum())


# 4. SELECTING DATA
print("\n--- Selecting Data ---")
print("Marks column:\n", df["Marks"])
print("Name and Marks:\n", df[["Name", "Marks"]])
print("Row 0 to 2 (loc, inclusive):\n", df.loc[0:2])
print("Avu's city:", df.loc[0, "City"])
print("First 2 rows, first 3 cols (iloc, exclusive):\n", df.iloc[0:2, 0:3])


# 5. FILTERING DATA
print("\n--- Filtering Data ---")
print("Marks above 75:\n", df[df["Marks"] > 75])
print("Grade A students:\n", df[df["Grade"] == "A"])
print("Age 21 AND Marks above 70:\n", df[(df["Age"] == 21) & (df["Marks"] > 70)])
print("Grade A OR Marks above 80:\n", df[(df["Grade"] == "A") | (df["Marks"] > 80)])
print("Not from Kannur:\n", df[df["City"] != "Kannur"])


# 6. ADDING / DROPPING COLUMNS
print("\n--- Adding/Dropping Columns ---")
df["Result"] = np.where(df["Marks"] >= 75, "Pass", "Fail")
df["Percentage"] = (df["Marks"] / 100) * 100
print("After adding Result & Percentage:\n", df)

df_dropped_col = df.drop("Percentage", axis=1)
print("After dropping Percentage column:\n", df_dropped_col)

df_dropped_row = df.drop(0, axis=0)
print("After dropping row 0:\n", df_dropped_row)


# 7. HANDLING MISSING VALUES
print("\n--- Handling Missing Values ---")
data2 = {
    "Name": ["Avu", "Chinnu", "Minnu", "Joy", "Oggy"],
    "Age": [21, np.nan, 20, 23, np.nan],
    "Marks": [85, 70, np.nan, 60, 75],
    "City": ["Kannur", "Kochi", None, "Kozhikode", "Trivandrum"]
}
df2 = pd.DataFrame(data2)
print("Data with missing values:\n", df2)
print("Missing values count:\n", df2.isnull().sum())
print("After dropna():\n", df2.dropna())

df2_filled = df2.fillna({
    "Age": df2["Age"].mean(),
    "Marks": df2["Marks"].mean(),
    "City": "Unknown"
})
print("After fillna() with mean/placeholder:\n", df2_filled)
print("After forward fill (ffill):\n", df2.ffill())


# 8. GROUPBY & AGGREGATION
print("\n--- Groupby & Aggregation ---")
data3 = {
    "Name": ["Avu", "Chinnu", "Minnu", "Joy", "Oggy", "Liya", "Anu"],
    "City": ["Kannur", "Kochi", "Kannur", "Kochi", "Kannur", "Kochi", "Kannur"],
    "Marks": [85, 70, 90, 60, 75, 95, 65],
    "Grade": ["A", "B", "A", "C", "B", "A", "C"]
}
df3 = pd.DataFrame(data3)
print(df3)
print("Average marks by city:\n", df3.groupby("City")["Marks"].mean())
print("Multiple stats by city:\n", df3.groupby("City")["Marks"].agg(["mean", "max", "min", "count"]))
print("Students per grade:\n", df3.groupby("Grade").size())
print("Group by City and Grade:\n", df3.groupby(["City", "Grade"])["Marks"].mean())


# 9. SORTING
print("\n--- Sorting ---")
print("Sorted by Marks (ascending):\n", df3.sort_values("Marks"))
print("Sorted by Marks (descending):\n", df3.sort_values("Marks", ascending=False))
print("Sorted by City then Marks:\n", df3.sort_values(["City", "Marks"], ascending=[True, False]))

df3["Rank"] = df3["Marks"].rank(ascending=False)
print("With rank:\n", df3.sort_values("Rank"))


# 10. MERGING & JOINING
print("\n--- Merging & Joining ---")
students_info = pd.DataFrame({
    "Name": ["Avu", "Chinnu", "Minnu", "Joy"],
    "Age": [21, 22, 20, 23]
})
students_marks = pd.DataFrame({
    "Name": ["Avu", "Chinnu", "Minnu", "Joy"],
    "Marks": [85, 70, 90, 60]
})
merged = pd.merge(students_info, students_marks, on="Name")
print("Merged data:\n", merged)

students_marks2 = pd.DataFrame({
    "Name": ["Avu", "Chinnu", "Liya"],
    "Marks": [85, 70, 95]
})
inner_merge = pd.merge(students_info, students_marks2, on="Name", how="inner")
print("Inner join:\n", inner_merge)

left_merge = pd.merge(students_info, students_marks2, on="Name", how="left")
print("Left join:\n", left_merge)


# 11. FEATURE ENGINEERING
print("\n--- Feature Engineering ---")
df4 = pd.DataFrame({
    "Name": ["Avu", "Chinnu", "Minnu", "Joy", "Oggy"],
    "Age": [21, 22, 20, 23, 21],
    "Marks": [85, 70, 90, 60, 75],
    "Study_Hours": [5, 3, 6, 2, 4]
})

df4["Marks_per_Hour"] = df4["Marks"] / df4["Study_Hours"]
print("With efficiency feature:\n", df4)

df4["Age_Group"] = pd.cut(df4["Age"], bins=[0, 20, 22, 25], labels=["Young", "Mid", "Senior"])
print("With age groups:\n", df4)

encoded = pd.get_dummies(df4["Age_Group"])
print("One-hot encoded:\n", encoded)

df4_final = pd.concat([df4, encoded], axis=1)
print("Final feature-engineered data:\n", df4_final)
