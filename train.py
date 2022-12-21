import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from datetime import datetime
import json
from mdutils.mdutils import MdUtils
import subprocess


def Model_Training(data,target_var):
    #, seed,test_size_p ,max_depth
    models_to_train = ['Random Forest']

    now = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    with open('train_config.json') as file:
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
    

    select_model = {
        'Random Forest': RandomForestRegressor(),
        'Gradient Boost': GradientBoostingClassifier()
        }
    model_train = select_model.get(model_type[1])
    n = models_to_train[0]

    pred_value = target_var

    pred_value_list = data[target_var]
       
    mdFile = MdUtils(file_name='Model_Metrics', title='Model Metrics')

    mdFile.new_header(level=1,title=n)
    release_num = 'Release1'
    build_num = '2021.1'
    mdFile.new_paragraph(f'This Markdown File contains the model metrics for the {release_num} and {build_num}')
    
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
        #regr = RandomForestRegressor(max_depth=6, random_state=seed)
        
        model_train.fit(X_train, y_train)
        # Report training set score
        regr_train_score = model_train.score(X_train, y_train) * 100

        regr_test_score = model_train.score(X_test, y_test) * 100

        # Write scores to a file
        

        title = f'{n}-Experiment {id}'

        mdFile.new_header(2,title)
        mdFile.new_line(f'{desc}')
        mdFile.new_line("Training variance explained: %2.1f%%" % regr_train_score)
        mdFile.new_line("Test variance explained: %2.1f%%" % regr_test_score)
        """
        with open(id+'_'+n+"_"+now+"_metrics.txt", 'w') as outfile:
            outfile.write(n+" Training variance explained: %2.1f%%\n" % regr_train_score)
            outfile.write(n+" Test variance explained: %2.1f%%\n" % regr_test_score)
        """
    
        model_importances = model_train.feature_importances_
        labels = df.columns


        list_of_strings = ["Feature", "Importance"]
        model_feature_df = pd.DataFrame(list(zip(labels,model_importances)), columns = ["feature","importance"])
        model_feature_df = model_feature_df.sort_values(by='importance', ascending=False,)
        
        model_feature_list = model_feature_df['feature'].tolist()
        model_importance_list = model_feature_df['importance'].tolist()
        #mdFile.new_list(labels)
        for f,b in zip(model_feature_list,model_importance_list):
            list_of_strings.extend([f,str(b)])
        col_names_len = len(["Feature", "Importance"])

        mdFile.new_table(columns=col_names_len, rows=len(model_importance_list)+1, text= list_of_strings)

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

        data[target_var] = pred_value_list



    mdFile.create_md_file()
    return print("Complete")


if __name__ == "__main__":

        df = pd.read_csv("winequality-red.csv")

        Model_Training(df,'quality')
