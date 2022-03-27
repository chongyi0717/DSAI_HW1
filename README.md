# DSAI-HW-2021
資料前處理：Robustscaler中位數和四分位數標準化,可以有效的縮放帶有outlier的數據，透過Robust如果數據中含有異常值在縮放中會捨去。  
模型训练资料：淨尖峰供電能力(MW)	尖峰負載(MW)	operating_reserve	備轉容量率(%)	used_power（工業用電+民生用電）	produced_power（所有發電站供電總和）  
模型訓練方式：使用tensorflow keras  
訓練時除了最後15筆資料都放入模型訓練，剩下15筆用來測試模型  
模型預測成效：  
![prediction](https://user-images.githubusercontent.com/49266509/160287958-ade076c4-0449-4833-baaa-2670a3232b23.png)  
程式執行：python app.py --training training_data.csv --output submission.csv  


