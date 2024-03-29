# Cat and Dog Classifier

## **About**

In this project, I build an algorithm, a deep learning model to classifier whether images contain either dog or cat. The output is not high accuracy because the main objective of this project is learning how to use convolution neural network (CNN) for classification.

## **Data**

The data is available at Kaggle. You can be download data using this link 
https://www.kaggle.com/tongpython/cat-and-dog

You should extract this data and save it in a folder called data

## **Installation**
  ### **Create Virtual Environment**
You should create a virtual environment using conda: 
````bash
conda create -n catdog python=3.8
````
```bash
conda activate catdog
````

### **Install dependencies** 
````bash
pip install -r requirements.txt
````

### **Download and Setup data**
````bash
bash setupdata.sh
````

## **Usage**

To execute the project, you only need to run the file below in your terminal

```bash
python main.py
```
You can use Streamlit API by:
````bash
streamlit run app.py
````

