## Multi Layer Peceptron electrocardiogram classification with Docker Deployment
#### A multi layer perceptron with 2 hidden layers of 64 and 32 neurons and sigmoid activation functions for a binary classification of patients 

1 - The project is structured as such : 
```
.
├── analysis.ipynb
├── app.dockerfile
├── app.py
├── dataset
│   └── ecg.csv
├── model.pkl
├── model.py
├── README.md
├── requirements.txt
├── scaler.pkl
└── train.dockerfile
```
## Building and execution of the containers :

### For the training : 

1 - Construct the docker image :
 
`docker build -f train.dockerfile -t ml-training . `

2 - Execute the docker container :

``` docker run -v $(pwd)/models:/app/models ml-training ```

### For the application : 

1 -  Construct the docker image :

``docker build -f app.dockerfile -t ml-api .``

2 - Execute the docker container : 

``docker run -p 5000:5000 ml-api``
