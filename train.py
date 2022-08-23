import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# Set random seed
seed = 42

################################
########## DATA PREP ###########
################################

# Load in the data
df = pd.read_csv("winequality-red.csv")

# Split into train and test sections
y = df.pop("quality")
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=seed)

#################################
########## MODELLING ############
#################################

# Fit a model on the train section
regr = RandomForestRegressor(max_depth=6, random_state=seed)
gbc = GradientBoostingClassifier(random_state=seed)



regr.fit(X_train, y_train)
gbc.fit(X_train, y_train)

# Report training set score
regr_train_score = regr.score(X_train, y_train) * 100
gbc_train_score = gbc.score(X_train, y_train) * 100
# Report test set score
regr_test_score = regr.score(X_test, y_test) * 100
gbc_test_score = gbc.score(X_test, y_test) * 100

# Write scores to a file
with open("metrics.txt", 'w') as outfile:
        outfile.write("Random Forest Training variance explained: %2.1f%%\n" % regr_train_score)
        outfile.write("Random Forest Test variance explained: %2.1f%%\n" % regr_test_score)
        outfile.write("GBC Training variance explained: %2.1f%%\n" % gbc_train_score)
        outfile.write("GBC Test variance explained: %2.1f%%\n" % gbc_test_score)



##########################################
##### PLOT FEATURE IMPORTANCE ############
##########################################
# Calculate feature importance in random forest
regr_importances = regr.feature_importances_
labels = df.columns
regr_feature_df = pd.DataFrame(list(zip(labels,regr_importances)), columns = ["feature","importance"])
regr_feature_df = regr_feature_df.sort_values(by='importance', ascending=False,)

# image formatting
axis_fs = 18 #fontsize
title_fs = 22 #fontsize
sns.set(style="whitegrid")

ax = sns.barplot(x="importance", y="feature", data=regr_feature_df)
ax.set_xlabel('Importance',fontsize = axis_fs) 
ax.set_ylabel('Feature', fontsize = axis_fs)#ylabel
ax.set_title('Random forest\nfeature importance', fontsize = title_fs)

plt.tight_layout()
plt.savefig("Random_Forest_feature_importance.png",dpi=120) 
plt.close()


# Calculate feature importance in random forest
gbc_importances = gbc.feature_importances_
labels = df.columns
gbc_feature_df = pd.DataFrame(list(zip(labels,gbc_importances)), columns = ["feature","importance"])
gbc_feature_df = gbc_feature_df.sort_values(by='importance', ascending=False,)

# image formatting
axis_fs = 18 #fontsize
title_fs = 22 #fontsize
sns.set(style="whitegrid")

ax = sns.barplot(x="importance", y="feature", data=gbc_feature_df)
ax.set_xlabel('Importance',fontsize = axis_fs) 
ax.set_ylabel('Feature', fontsize = axis_fs)#ylabel
ax.set_title('Gradient Boost\nfeature importance', fontsize = title_fs)

plt.tight_layout()
plt.savefig("GBC_feature_importance.png",dpi=120) 
plt.close()









##########################################
############ PLOT RESIDUALS  #############
##########################################

y_pred = regr.predict(X_test) + np.random.normal(0,0.25,len(y_test))
y_jitter = y_test + np.random.normal(0,0.25,len(y_test))
res_df = pd.DataFrame(list(zip(y_jitter,y_pred)), columns = ["true","pred"])

ax = sns.scatterplot(x="true", y="pred",data=res_df)
ax.set_aspect('equal')
ax.set_xlabel('True wine quality',fontsize = axis_fs) 
ax.set_ylabel('Predicted wine quality', fontsize = axis_fs)#ylabel
ax.set_title('Residuals', fontsize = title_fs)

# Make it pretty- square aspect ratio
ax.plot([1, 10], [1, 10], 'black', linewidth=1)
plt.ylim((2.5,8.5))
plt.xlim((2.5,8.5))

plt.tight_layout()
plt.savefig("Random Forest residuals.png",dpi=120) 




y_pred = gbc.predict(X_test) + np.random.normal(0,0.25,len(y_test))
y_jitter = y_test + np.random.normal(0,0.25,len(y_test))
res_df = pd.DataFrame(list(zip(y_jitter,y_pred)), columns = ["true","pred"])

ax = sns.scatterplot(x="true", y="pred",data=res_df)
ax.set_aspect('equal')
ax.set_xlabel('True wine quality',fontsize = axis_fs) 
ax.set_ylabel('Predicted wine quality', fontsize = axis_fs)#ylabel
ax.set_title('Residuals', fontsize = title_fs)

# Make it pretty- square aspect ratio
ax.plot([1, 10], [1, 10], 'black', linewidth=1)
plt.ylim((2.5,8.5))
plt.xlim((2.5,8.5))

plt.tight_layout()
plt.savefig("GBC residuals.png",dpi=120) 