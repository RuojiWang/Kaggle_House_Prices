# Kaggle_House_Prices

* Kaggle_House_Prices is a semi-automatic deep learning experiment code for kaggle house prices competition, which is a good example for beginners with regression problems.
* Semi-automatic means autimaic hyperparameters optimization including compute and save the best model in history and output prediction file and intermediate computation file, which can save you lots of time. 
* Kaggle_House_Prices treats network structure design and network initialization and so on as hyperparameters selection. And it supports deep learning model, which allows you to achieve better accuracy than traditional model like xgboost easily. 
* Besides determining selection of hyperparameters for optimization, you only work is a little feature engineering like feature scale or little code change like file path. 
* Just enjoy deep learning and build your solution based on Kaggle_House_Prices for other competitions.
<br></br>

### Environment Configuration

* Anaconda pytorch skorch hyperopt and python 3.6 are mainly needed. 
* You can use the following commands: 
  * pip install anaconda
  * pip install pytorch
  * pip install -u skorch
  * pip install hyperopt
<br></br>

### Running on House Prices dataset

* Firstly, configure the running environment.
* Secondly, run examples and learn a little about pytorch skorch and hyperopt.
* Thirdly, include all the files into your project and copy the files in house_prices_files.rar, which are the raw data for kaggle house prices competiton. And then replace the pd.read_csv path with your copy path. 
* Finally, you can run and debug the auto_model_example.py to learn more details. 
<br></br>

### Using for Other Competition
  
* Firstly, configure the running environment.
* Secondly, run the auto_model_example.py on the house prices dataset to learn details.
* Thirdly, do little feature engineering and hyperparameters selection for new competiton. Which means space space_nodes best_nodes and parse_space funciton in the code are mainly needed to modified. **Remeber to modify space space_nodes best_nodes and parse_space function at the same time!**
* Finally, modify the auto_model_example.py or optimize your neural network model until you get satisfying results. Whenever you train the model, the best hyperparameters in history and prediction file of the model will be saved. 
<br></br>

### Contact Information

Any question please contact the following email:
* 1035456235@qq.com
* YeaTmeRet@gmail.com
  
