import pandas as pd
from sklearn.model_selection import train_test_split

# Load raw Titanic data
df = pd.read_csv("titanic.csv")

# Select and rename columns
df = df[['Survived','Sex','Age','SibSp','Parch','Fare','Pclass','Cabin','Embarked']]
df.columns = [
    'survived','sex','age','n_siblings_spouses','parch',
    'fare','class','deck','embark_town'
]

# Feature engineering
df['deck'] = df['deck'].str[0].fillna('Unknown')

df['embark_town'] = df['embark_town'].map({
    'S': 'Southampton',
    'C': 'Cherbourg',
    'Q': 'Queenstown'
})

df['alone'] = (df['n_siblings_spouses'] + df['parch'] == 0)

# Remove rows with missing age (optional)
df = df.dropna()

# Split dataset (80% train, 20% eval)
train_df, eval_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

# Save CSV files
train_df.to_csv("train.csv", index=False)
eval_df.to_csv("eval.csv", index=False)

print("Saved train.csv and eval.csv with 'survived'")
print("Train shape:", train_df.shape)
print("Eval shape:", eval_df.shape)
