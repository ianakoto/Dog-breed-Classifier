# Dog-breed-Classifier
A classifier to predict dogs when it detects a dog in an image and also  provides an estimate of the dog breed that is most resembling  when it detects a human in an image


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.


### Machine Learning Model

A Vgg16 and Resnet50 Pretained model was used for building the model

```
download and open the report.hml to get all codes and procedures for building the model

```



#### Web App

### Prerequisites

What things you need to install the software and how to install them

```
install visual studio code
install python 3.7.3
install node
install ionic

```

### Installing


Front end development env

```

run npm install  in the web\dogbreed-web-app\src directory to install the node modules


```

Backend development env

```
run pip install -r requirements.txt in the web\backend\src directory to install python modules

```


## Running the app



### Backend


```
navigate to  web\backend\src

run flask run to run api
```

### Front end


```
navigate to  web\dogbreed-web-app\src

run ionic serve to run pp
```

## Deployment
Deploying  flask restapi using Heroku

```
Create an account in heroku
Create an app in heroku

$ heroku login
$ heroku git:clone -a <app-name-on-heroku>
$ git add .
$ git commit -am "make it better"
$ git push heroku master
```




## Authors

* **Ian Cecil Mawuli Akoto** - *Initial work* - [ian0549](https://github.com/ian0549)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* udacity
* etc
