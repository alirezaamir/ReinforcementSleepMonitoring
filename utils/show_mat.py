import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np

conf_Mat=     [[314269,   1411,  14738 ,   626 ,  8457],
 [ 10054  , 1345 , 13008 ,    18 ,  7621],
 [ 17349  , 1600 ,273216 , 15457 , 16409],
 [   810  ,    2 , 20120 , 68832 ,    32],
          [ 15069 ,   704 , 19329  ,  171  ,82098]]

sum = np.sum(conf_Mat,axis=1)
for i in range(5):
    for j in range(5):
        conf_Mat[i][j] = conf_Mat[i][j] / sum[i]

df_cm = pd.DataFrame(conf_Mat, index = ["Wake", "N1", "N2", "N3", "REM"],
                  columns = ["Wake", "N1", "N2", "N3", "REM"])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)
plt.savefig('../outputs/fig/conf.png')


