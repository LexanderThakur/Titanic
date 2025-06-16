import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")
# Load and train inside app (for simplicity)
@st.cache_data
def load_and_train():
    df = pd.read_csv("titanic_feature_5.csv")

    def group_age(x):
        if x < 12:
            return 0
        elif x < 20:
            return 1
        elif x < 35:
            return 2
        elif x < 60:
            return 3
        else:
            return 4

    df['Age_Group2'] = df['Age'].apply(group_age)

    features = ['Pclass', 'Sex_numerical', 'Age_Group2', 'Embarked_encoded', 'is_unknown', 'family_size']
    X = df[features]
    y = df['Survived']

    corr_matrix = df[features + ['Survived']].corr()

    model = LogisticRegression()
    model.fit(X, y)
    return model, corr_matrix


model, corr_matrix = load_and_train()


# ---- Streamlit App ----


tab1, tab2 = st.tabs(["📘 Project Workflow", "🧮 Predict Survival"])

# Tab 1 - Workflow
with tab1:
    st.title("Titanic Survival Project Overview")
    st.markdown("""
    ### 🚢 Dataset
    Titanic passenger dataset including age, gender, family, and travel class, our goal is to predict who survived.  \n
    Check out for yourself at link [Kaggle](https://www.kaggle.com/competitions/titanic)
                


    ### Prepration
    - Filled 177 missing age values with median.
    - Filled 2 missing Embarked values with mode.
    - Marked 687 Cabin values as unknown.
    Code snippet:
    """)
    st.code("""
titanic['Age'].fillna(titanic['Age'].median(),inplace=True)
titanic['Embarked'].fillna(titanic['Embarked'].mode()[0],inplace=True)
titanic['Cabin'].fillna('Unknown',inplace=True)
titanic.to_csv('titanic_filled.csv', index=False)
""", language='python')
   
    st.markdown("""
    ### Feature Engineering
    - Converted gender to numerical value.
    - Grouped ages numerically according to youth.
    - Encoded Embarked Values.
    - Added a is_unkown feature if cabin not known.
    - Added a family size feature
""")
    st.code("""
titanic['Sex_numerical']= titanic['Sex'].apply(lambda x:1 if x=='male' else 0)
def group_age(x):
    if x < 12:
        return 0  # Child
    elif x < 20:
        return 1  # Teen
    elif x < 35:
        return 2  # Young adult
    elif x < 60:
        return 3  # Adult
    else:
        return 4  # Senior
titanic['Age_Group2']= titanic['Age'].apply(lambda x: group_age(x))
titanic['Embarked_encoded'] = le1.fit_transform(titanic['Embarked'])
titanic['family_size']=titanic['SibSp']+titanic['Parch']
""", language='python')
    
    st.markdown("""
    ### Feature Intuition
    - Age helps as childern and elders are less likely to survive.
    - Gender tell about physical capabilities and rescue effforts priority.
    - Unkown cabin implies not survived.
    - Passanger Class is use full as how rich a passanger is might matter.
    - Family_size is used as smaller families would we more likely to survive.  \n
    Correlation matrix:
                

""")

    st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm').format(precision=2))


    st.markdown("""
    ### Linear Discriminant Analysis Model
    - Accuracy on test set 77.27%
    - Accuracy via cross validation with k=5 80.20%  \n
    

""")
    st.markdown("### 📊 Bayes’ Theorem for Classification")
    st.latex(r'''
P(y = k \mid x) = \frac{P(x \mid y = k) \cdot P(y = k)}{P(x)} 
= \frac{P(x \mid y = k) \cdot P(y = k)}{\sum_l P(x \mid y = l) \cdot P(y = l)}
''')

    st.code("""
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
model=LinearDiscriminantAnalysis()
titanic=pd.read_csv('titanic_feature_5.csv');
test=pd.read_csv('./test_feature_5.csv')

titanic['Age_Group2']= titanic['Age'].apply(lambda x: group_age(x))
test['Age_Group2']=test['Age'].apply(lambda x: group_age(x))


X=titanic[['Pclass','Sex_numerical','Age_Group2','Embarked_encoded','is_unknown','family_size']]
Y=titanic['Survived']

model.fit(X,Y)

X_test=test[['Pclass','Sex_numerical','Age_Group2','Embarked_encoded','is_unknown','family_size']]

prediction_3=pd.DataFrame(columns=['PassengerId','Survived'])

prediction_3['PassengerId']=test['PassengerId']
prediction_3['Survived']=model.predict(X_test)

prediction_3.to_csv('prediction_final.csv',index=False)
""", language='python')



    # ### 📈 Model
    # Logistic Regression  
    # Accuracy: ~77.2% (Cross-validation)

    # ### 🎯 Goal
    # Predict if a Titanic passenger would survive based on their info.
    # """)

# Tab 2 - Interactive Predictor
with tab2:
    st.title("🧮 Titanic Survival Predictor")

    st.markdown("Fill in the passenger details:")

    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    sex = st.radio("Sex", ['Male', 'Female'])
    age_group = st.selectbox("Age Group", [
        'Child (<12)', 'Teen (12–20)', 'Young Adult (20–35)', 'Adult (35–60)', 'Senior (60+)'
    ])
    embarked = st.selectbox("Embarked From", ['S', 'C', 'Q'])
    is_unknown = st.checkbox("Cabin Unknown?", value=True)
    family_size = st.slider("Family Size", min_value=1, max_value=10, value=1)

    # Mapping to model input
    sex_num = 0 if sex == 'Male' else 1
    age_map = {'Child (<12)': 0, 'Teen (12–20)': 1, 'Young Adult (20–35)': 2, 'Adult (35–60)': 3, 'Senior (60+)': 4}
    embarked_map = {'S': 2, 'C': 0, 'Q': 1}

    if st.button("Predict"):
        input_data = pd.DataFrame([[pclass, sex_num, age_map[age_group],
                                     embarked_map[embarked], 1 if is_unknown else 0,
                                     family_size]],
                                   columns=['Pclass', 'Sex_numerical', 'Age_Group2',
                                            'Embarked_encoded', 'is_unknown', 'family_size'])

        result = model.predict(input_data)[0]

        if result == 1:
            st.success("🎉 Survived!")
            st.image("survived_meme.png", caption="Lucky you!", use_container_width=True)
        else:
            st.error("☠️ Did not survive.")
            st.image("not_survived_meme.jpg", caption="Better luck next time...", use_container_width=True)

