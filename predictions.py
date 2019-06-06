import pickle
import pandas as pd
import datetime as dt

if __name__ == '__main__':
    df = pd.read_csv('titanic_test.csv')
    model = pickle.load(open('final_model.pkl', 'rb'))
    
    # drop unnecessary columns
    df.drop(['PassengerId','Name','Ticket','Cabin'], axis = 1, inplace = True)
    
    # encode 'Embarked' col
    df = pd.concat([df, pd.get_dummies(df['Embarked'])], axis=1)
    df.drop('Embarked', axis = 1, inplace = True)
    df.columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_C', 'Embarked_Q','Embarked_S']
    
    # map '1' to males and '0' to females
    df['Sex'] = df['Sex'].apply(lambda x: 1 if x == 'male' else 0)
    
    # generate predictions
    df['survival_pred'] = model.predict(df[['Pclass','Sex','Embarked_C','Embarked_S']])

    # output to csv
    df.to_csv('model_predictions_{0}.csv'.format(dt.date.today().strftime('%Y-%m-%d')))