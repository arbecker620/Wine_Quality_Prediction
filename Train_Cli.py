import typer
import subprocess
from datetime import datetime
import json
import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np




app = typer.Typer()


@app.command()
def hello(name: str):
    print(f"Hello {name}")


@app.command()
def goodbye(name: str, formal: bool = False):
    if formal:
        print(f"Goodbye Ms. {name}. Have a good day.")
    else:
        print(f"Bye {name}!")

@app.command()
def Model_Training_cli(data_path, training_json_path):
    """
    Provide a dataset to train a ML Model from.

    """
    now = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    with open(training_json_path) as file:
        json_df = json.load(file)
    
    
    list_values=json_df.get('Experiment')
    
    model_type = []
    id = []
    seed = []
    training_set = []
    description = []

    col_name = ['ID', 'Seed', 'Training Set', 'Model Type', 'Description']
    for n in list_values:
        id.append(n['ID'])
        seed.append(n['Seed'])
        training_set.append(n['Training Set']/100)
        model_type.append(n['Model Type'])
        description.append(n['Description'])
    
    experiments_df=pd.DataFrame(list(zip(id, seed, training_set,model_type,description)),
              columns=col_name)

    data = pd.read_csv(data_path)
    
    pred_value = 'quality'


    pred_value_list = data[pred_value]    
    for index, row in experiments_df.iterrows():
        seed = row['Seed']
        n = row['Model Type']
        id = str(row['ID'])
        desc = row['Description']
        select_model = {
        'Random Forest': RandomForestRegressor(),
        'Gradient Boost': GradientBoostingClassifier()
        }

        model_train = select_model.get(n)
        y = data.pop(pred_value)
        X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=row['Training Set'], random_state=row['Seed'])
        
        model_train.fit(X_train, y_train)
        # Report training set score
        regr_train_score = model_train.score(X_train, y_train) * 100

        regr_test_score = model_train.score(X_test, y_test) * 100

        # Write scores to a file
        

        print("Training variance explained: %2.1f%%" % regr_train_score)
        print("Test variance explained: %2.1f%%" % regr_test_score)

    
        model_importances = model_train.feature_importances_
        labels = data.columns


        list_of_strings = ["Feature", "Importance"]
        model_feature_df = pd.DataFrame(list(zip(labels,model_importances)), columns = ["feature","importance"])
        model_feature_df = model_feature_df.sort_values(by='importance', ascending=False,)
        
        model_feature_list = model_feature_df['feature'].tolist()
        model_importance_list = model_feature_df['importance'].tolist()
        #mdFile.new_list(labels)
        for f,b in zip(model_feature_list,model_importance_list):
            list_of_strings.extend([f,str(b)])
        col_names_len = len(["Feature", "Importance"])

        print(model_feature_df)

        axis_fs = 18 #fontsize
        title_fs = 22 #fontsize
        sns.set(style="whitegrid")
        ax = sns.barplot(x="importance", y="feature", data=model_feature_df)
        ax.set_xlabel('Importance',fontsize = axis_fs) 
        ax.set_ylabel('Feature', fontsize = axis_fs)#ylabel
        ax.set_title(n+'\nfeature importance', fontsize = title_fs)
        plt.tight_layout()
        #plt.savefig(id+'_'+n+"_"+now+"_feature_importance.png",dpi=120) 
        #plt.close()
        plt.show()

        data[pred_value] = pred_value_list
        



    return print("Complete")


@app.command()
def List_Packages_CLi():
    """
    This provides a list of all packages installed with this CLI app.

    """
    print("here is a list")



if __name__ == "__main__":
    app()



